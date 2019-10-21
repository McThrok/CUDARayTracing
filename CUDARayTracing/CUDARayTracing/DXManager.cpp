#include "DXManager.h"

#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

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


////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
bool DXManager::DrawScene()
{
	// Clear the backbuffer to a black color
	float ClearColor[4] = { 0.5f, 0.5f, 0.6f, 1.0f };
	g_pd3dDeviceContext->ClearRenderTargetView(g_pSwapChainRTV, ClearColor);

	float quadRect[4] = { -1,  -1,  2 , 2 };
	//
	// draw the 2d texture
	//
	HRESULT hr;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	ConstantBuffer* pcb;
	hr = g_pd3dDeviceContext->Map(g_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	AssertOrQuit(SUCCEEDED(hr));
	pcb = (ConstantBuffer*)mappedResource.pData;
	{
		memcpy(pcb->vQuadRect, quadRect, sizeof(float) * 4);
	}
	g_pd3dDeviceContext->Unmap(g_pConstantBuffer, 0);
	g_pd3dDeviceContext->Draw(4, 0);

	// Present the backbuffer contents to the display
	g_pSwapChain->Present(0, 0);
	return true;
}

void DXManager::Cleanup()
{
		if (g_pVertexShader)
			g_pVertexShader->Release();

		if (g_pPixelShader)
			g_pPixelShader->Release();

		if (g_pConstantBuffer)
			g_pConstantBuffer->Release();

		if (g_pSamplerState)
			g_pSamplerState->Release();

		if (g_pSwapChainRTV != nullptr)
			g_pSwapChainRTV->Release();

		if (g_pSwapChain != nullptr)
			g_pSwapChain->Release();

		if (g_pd3dDevice != nullptr)
			g_pd3dDevice->Release();
}