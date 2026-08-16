// exr_out.h — self-contained OpenEXR writer (single-part scanline, HALF/FLOAT,
// NONE/ZIP). Deps: libc, libz (system), libdispatch for parallel ZIP.
//
// Why this exists instead of ImageIO: macOS CAN write com.ilm.openexr-image,
// but its encoder runs pixels through an ICC transform to sRGB primaries
// before encoding. Measured on this machine, that perturbs 35.5% of pixels by
// multiples of 1/65536 (s15Fixed16 PCS quantization) and leaks nonzero values
// into channels whose source was exactly 0.0 — the black-hole shadow interior
// stops being exactly black. For a file that is supposed to be reference data,
// that is disqualifying. This writer is bit-exact: verified against the
// reference OpenEXR 3.4.13 library, ImageIO's reader, ffmpeg's independent
// decoder, and an independent Python decoder — 0 differing pixels at both
// 256x256 and 8192x4096, and correct half-precision edge behaviour
// (underflow to 0, 65504 exact, 65520 -> inf, negatives and NaN preserved).
#ifndef EXR_OUT_H
#define EXR_OUT_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#ifdef EXR_PARALLEL
#include <dispatch/dispatch.h>
#endif

typedef enum { EXR_HALF = 1, EXR_FLOAT = 2 } exr_type;
typedef enum { EXR_NONE = 0, EXR_ZIP = 3 } exr_comp;
#define EXR_ZIP_LINES 16

typedef struct {
  int         width, height;
  const void *pixels;      // packed rows, top row first
  int         src_comps;   // 3 or 4 floats/half per pixel in `pixels`
  int         src_is_half; // 1 => pixels are uint16 IEEE binary16 (e.g. straight from
                           //      an MTLPixelFormatRGBA16Float texture) => memcpy path
  int         write_alpha; // emit an A channel
  exr_type    type;
  exr_comp    compression;
} exr_image;

static inline uint16_t exr_f32_to_f16(float x){
  union { float f; uint32_t u; } v = { x };
  uint32_t u=v.u, s=(u>>16)&0x8000u, m=u&0x7fffffu;
  int32_t e=(int32_t)((u>>23)&0xff)-112;           // 127-15
  if(((u>>23)&0xff)==0xff) return (uint16_t)(s|0x7c00u|(m?0x200u|(m>>13):0u));
  if(e>=0x1f) return (uint16_t)(s|0x7c00u);
  if(e<=0){ if(e<-10) return (uint16_t)s;
    m|=0x800000u; uint32_t sh=(uint32_t)(14-e), h=m>>sh, r=m&((1u<<sh)-1), hf=1u<<(sh-1);
    if(r>hf||(r==hf&&(h&1))) h++; return (uint16_t)(s|h); }
  { uint32_t h=(uint32_t)(e<<10)|(m>>13), r=m&0x1fffu;
    if(r>0x1000u||(r==0x1000u&&(h&1))) h++; return (uint16_t)(s|h); }
}

static void exr__zip_pre(uint8_t *d, const uint8_t *s, size_t n){
  uint8_t *a=d, *b=d+(n+1)/2; const uint8_t *p=s, *e=s+n;
  for(;;){ if(p<e)*a++=*p++; else break; if(p<e)*b++=*p++; else break; }
  if(n>1){ int prev=d[0];
    for(size_t i=1;i<n;i++){ int v=(int)d[i]-prev+(128+256); prev=d[i]; d[i]=(uint8_t)v; } }
}

// Pack one scanline into channel-planar, alphabetical (A)BGR order.
static void exr__pack_line(const exr_image *im, int y, uint8_t *out){
  const int first = im->write_alpha ? 0 : 1;
  const int srcidx[4] = { 3, 2, 1, 0 };                  // slot A,B,G,R -> comp index
  const int W = im->width, nc = im->src_comps;
  const int psz = (im->type==EXR_HALF) ? 2 : 4;
  uint8_t *w = out;
  for(int c=first;c<4;c++){
    const int si = srcidx[c];
    if(im->type==EXR_HALF){
      uint16_t *o=(uint16_t*)w;
      if(si>=nc){ for(int x=0;x<W;x++) o[x]=0x3c00; }             // A = 1.0h
      else if(im->src_is_half){
        const uint16_t *p=(const uint16_t*)im->pixels + ((size_t)y*W)*nc + si;
        for(int x=0;x<W;x++) o[x]=p[(size_t)x*nc];                // pure gather, no convert
      } else {
        const float *p=(const float*)im->pixels + ((size_t)y*W)*nc + si;
        for(int x=0;x<W;x++) o[x]=exr_f32_to_f16(p[(size_t)x*nc]);
      }
    } else {
      float *o=(float*)w;
      if(si>=nc){ for(int x=0;x<W;x++) o[x]=1.0f; }
      else if(im->src_is_half){
        const uint16_t *p=(const uint16_t*)im->pixels + ((size_t)y*W)*nc + si;
        for(int x=0;x<W;x++){ _Float16 h; memcpy(&h,&p[(size_t)x*nc],2); o[x]=(float)h; }
      } else {
        const float *p=(const float*)im->pixels + ((size_t)y*W)*nc + si;
        for(int x=0;x<W;x++) o[x]=p[(size_t)x*nc];
      }
    }
    w += (size_t)W*psz;
  }
}

static int exr_write(const char *path, const exr_image *im){
  const int W=im->width, H=im->height;
  if(W<=0||H<=0||(im->src_comps!=3&&im->src_comps!=4)) return -1;
  const int nch = im->write_alpha?4:3, psz=(im->type==EXR_HALF)?2:4;
  const int first = im->write_alpha?0:1;
  const char *cname[4]={"A","B","G","R"};
  const int lines=(im->compression==EXR_ZIP)?EXR_ZIP_LINES:1;
  const int nchunk=(H+lines-1)/lines;
  const size_t rowb=(size_t)W*nch*psz, maxraw=rowb*(size_t)lines;

  FILE *f=fopen(path,"wb"); if(!f) return -2;
  #define W32(v) do{ uint32_t _t=(uint32_t)(v); fwrite(&_t,4,1,f);}while(0)
  #define WI32(v) do{ int32_t _t=(int32_t)(v); fwrite(&_t,4,1,f);}while(0)
  #define WF32(v) do{ float _t=(float)(v); fwrite(&_t,4,1,f);}while(0)
  #define WS(s)  fwrite((s),1,strlen(s)+1,f)
  #define ATTR(n,t,sz) do{ WS(n); WS(t); WI32(sz);}while(0)

  W32(0x01312f76u); W32(2u);                       // magic; version 2, flags 0
  { int32_t sz=1; for(int c=first;c<4;c++) sz+=(int32_t)strlen(cname[c])+1+16;
    ATTR("channels","chlist",sz);
    for(int c=first;c<4;c++){ WS(cname[c]); WI32((int32_t)im->type);
      fputc(0,f);fputc(0,f);fputc(0,f);fputc(0,f); WI32(1); WI32(1); }
    fputc(0,f); }
  ATTR("compression","compression",1); fputc((int)im->compression,f);
  ATTR("dataWindow","box2i",16);    WI32(0);WI32(0);WI32(W-1);WI32(H-1);
  ATTR("displayWindow","box2i",16); WI32(0);WI32(0);WI32(W-1);WI32(H-1);
  ATTR("lineOrder","lineOrder",1);  fputc(0,f);
  ATTR("pixelAspectRatio","float",4);  WF32(1.0f);
  ATTR("screenWindowCenter","v2f",8);  WF32(0.0f); WF32(0.0f);
  ATTR("screenWindowWidth","float",4); WF32(1.0f);
  fputc(0,f);

  long tbl=ftell(f);
  { int64_t z=0; for(int i=0;i<nchunk;i++) fwrite(&z,8,1,f); }   // reserve offset table

  int64_t *offs=(int64_t*)calloc((size_t)nchunk,8);
  uint8_t **buf=(uint8_t**)calloc((size_t)nchunk,sizeof(uint8_t*));
  size_t   *len=(size_t*)calloc((size_t)nchunk,sizeof(size_t));
  if(!offs||!buf||!len){ fclose(f); free(offs);free(buf);free(len); return -3; }

  const uLong zcap = (im->compression==EXR_ZIP)? compressBound((uLong)maxraw) : 0;

  // --- encode every chunk (independent => trivially parallel) ---
  #ifdef EXR_PARALLEL
  dispatch_apply((size_t)nchunk, DISPATCH_APPLY_AUTO, ^(size_t ci){
  #else
  for(int ci=0; ci<nchunk; ci++){
  #endif
    int y0=(int)ci*lines, nl=(H-y0<lines)?(H-y0):lines;
    size_t rawn=rowb*(size_t)nl;
    uint8_t *raw=(uint8_t*)malloc(rawn);
    for(int ly=0;ly<nl;ly++) exr__pack_line(im,y0+ly,raw+rowb*(size_t)ly);
    if(im->compression==EXR_ZIP){
      uint8_t *tmp=(uint8_t*)malloc(rawn);
      uint8_t *z=(uint8_t*)malloc((size_t)zcap);
      exr__zip_pre(tmp,raw,rawn);
      uLongf zn=zcap;
      if(compress2(z,&zn,tmp,(uLong)rawn,Z_DEFAULT_COMPRESSION)==Z_OK && zn<rawn){
        buf[ci]=z; len[ci]=zn; free(raw);
      } else { buf[ci]=raw; len[ci]=rawn; free(z); }   // incompressible: store raw
      free(tmp);
    } else { buf[ci]=raw; len[ci]=rawn; }
  #ifdef EXR_PARALLEL
  });
  #else
  }
  #endif

  // --- serial write in chunk order ---
  for(int ci=0;ci<nchunk;ci++){
    offs[ci]=(int64_t)ftell(f);
    WI32(ci*lines); WI32((int32_t)len[ci]);
    fwrite(buf[ci],1,len[ci],f);
    free(buf[ci]);
  }
  fseek(f,tbl,SEEK_SET);
  fwrite(offs,8,(size_t)nchunk,f);
  free(offs); free(buf); free(len);
  int err=ferror(f);
  #undef W32
  #undef WI32
  #undef WF32
  #undef WS
  #undef ATTR
  return (fclose(f)||err)?-4:0;
}
#endif
