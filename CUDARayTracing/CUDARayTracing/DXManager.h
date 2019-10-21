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


	bool findDXDevice(char* dev_name)
	{
		HRESULT hr = S_OK;
		cudaError cuStatus;

		// Iterate through the candidate adapters
		IDXGIFactory* pFactory;
		hr = sFnPtr_CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory));

		if (!SUCCEEDED(hr))
		{
			printf("> No DXGI Factory created.\n");
			return false;
		}

		UINT adapter = 0;

		for (; !g_pCudaCapableAdapter; ++adapter)
		{
			// Get a candidate DXGI adapter
			IDXGIAdapter* pAdapter = nullptr;
			hr = pFactory->EnumAdapters(adapter, &pAdapter);

			if (FAILED(hr))
			{
				break;    // no compatible adapters found
			}

			// Query to see if there exists a corresponding compute device
			int cuDevice;
			cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
			printLastCudaError("cudaD3D11GetDevice failed"); //This prints and resets the cudaError to cudaSuccess

			if (cudaSuccess == cuStatus)
			{
				// If so, mark it as the one against which to create our d3d10 device
				g_pCudaCapableAdapter = pAdapter;
				g_pCudaCapableAdapter->AddRef();
			}

			pAdapter->Release();
		}

		printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

		pFactory->Release();

		if (!g_pCudaCapableAdapter)
		{
			printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
			return false;
		}

		DXGI_ADAPTER_DESC adapterDesc;
		g_pCudaCapableAdapter->GetDesc(&adapterDesc);
		wcstombs(dev_name, adapterDesc.Description, 128);

		printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
		printf("> %s\n", dev_name);

		return true;
	}

	void Cleanup();
	bool DrawScene();
};

