#pragma once

#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

struct ConstantBuffer
{
	float   vQuadRect[4];
};

class DXManager
{
public:
	int width;
	int height;

	// Data structure for 2D texture shared between DX10 and CUDA
	ID3D11Texture2D* pTexture;
	ID3D11ShaderResourceView* pSRView;
	int offsetInShader;

	IDXGIAdapter* g_pCudaCapableAdapter = nullptr;  // Adapter to use
	ID3D11Device* g_pd3dDevice = nullptr; // Our rendering device
	ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
	IDXGISwapChain* g_pSwapChain = nullptr; // The swap chain of the window
	ID3D11RenderTargetView* g_pSwapChainRTV = nullptr; //The Render target view on the swap chain ( used for clear)
	ID3D11RasterizerState* g_pRasterState = nullptr;

	static const char g_simpleShaders[];

	ID3D11VertexShader* g_pVertexShader;
	ID3D11PixelShader* g_pPixelShader;
	ID3D11Buffer* g_pConstantBuffer;
	ID3D11SamplerState* g_pSamplerState;

	HRESULT InitTextures();
	void Cleanup();
	bool DrawScene();


	HRESULT InitD3D(HWND hWnd);

	bool findDXDevice();
	bool loadDevice();
};

