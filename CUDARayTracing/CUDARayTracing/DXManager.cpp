#include "DXManager.h"

const char DXManager::g_simpleShaders[] = " \n" \
"cbuffer cbuf  \n" \
"{  \n" \
"  float4 g_vQuadRect;  \n" \
"}  \n" \
"Texture2D g_Texture2D;  \n" \
" \n" \
"SamplerState samLinear{  \n" \
"    Filter = MIN_MAG_LINEAR_MIP_POINT;  \n" \
"}; \n" \
" \n" \
"struct Fragment{  \n" \
"    float4 Pos : SV_POSITION; \n" \
"    float3 Tex : TEXCOORD0; }; \n" \
" \n" \
"Fragment VS( uint vertexId : SV_VertexID ) \n" \
"{ \n" \
"    Fragment f; \n" \
"    f.Tex = float3( 0.f, 0.f, 0.f);  \n" \
"    if (vertexId == 1) f.Tex.x = 1.f;  \n" \
"    else if (vertexId == 2) f.Tex.y = 1.f;  \n" \
"    else if (vertexId == 3) f.Tex.xy = float2(1.f, 1.f);  \n" \
"     \n" \
"    f.Pos = float4( g_vQuadRect.xy + f.Tex * g_vQuadRect.zw, 0, 1); \n" \
"     \n" \
"    if (vertexId == 1) f.Tex.z = 0.5f;  \n" \
"    else if (vertexId == 2) f.Tex.z = 0.5f;  \n" \
"    else if (vertexId == 3) f.Tex.z = 1.f;  \n" \
" \n" \
"    return f; \n" \
"} \n" \
" \n" \
"float4 PS( Fragment f ) : SV_Target \n" \
"{ \n" \
"    return g_Texture2D.Sample( samLinear, f.Tex.xy ); \n" \
"} \n" \
"\n";