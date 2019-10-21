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

	HRESULT InitD3D(HWND hWnd)
	{
		HRESULT hr = S_OK;
		cudaError cuStatus;

		// Set up the structure used to create the device and swapchain
		DXGI_SWAP_CHAIN_DESC sd;
		ZeroMemory(&sd, sizeof(sd));
		sd.BufferCount = 1;
		sd.BufferDesc.Width = width;
		sd.BufferDesc.Height = height;
		sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		sd.BufferDesc.RefreshRate.Numerator = 60;
		sd.BufferDesc.RefreshRate.Denominator = 1;
		sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		sd.OutputWindow = hWnd;
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
		sd.Windowed = TRUE;

		D3D_FEATURE_LEVEL tour_fl[] =
		{
			D3D_FEATURE_LEVEL_11_0,
			D3D_FEATURE_LEVEL_10_1,
			D3D_FEATURE_LEVEL_10_0
		};
		D3D_FEATURE_LEVEL flRes;
		// Create device and swapchain
		hr = sFnPtr_D3D11CreateDeviceAndSwapChain(
			g_pCudaCapableAdapter,
			D3D_DRIVER_TYPE_UNKNOWN,//D3D_DRIVER_TYPE_HARDWARE,
			nullptr, //HMODULE Software
			0, //UINT Flags
			tour_fl, // D3D_FEATURE_LEVEL* pFeatureLevels
			3, //FeatureLevels
			D3D11_SDK_VERSION, //UINT SDKVersion
			&sd, // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
			&g_pSwapChain, //IDXGISwapChain** ppSwapChain
			&g_pd3dDevice, //ID3D11Device** ppDevice
			&flRes, //D3D_FEATURE_LEVEL* pFeatureLevel
			&g_pd3dDeviceContext//ID3D11DeviceContext** ppImmediateContext
		);
		AssertOrQuit(SUCCEEDED(hr));

		g_pCudaCapableAdapter->Release();

		// Get the immediate DeviceContext
		g_pd3dDevice->GetImmediateContext(&g_pd3dDeviceContext);

		// Create a render target view of the swapchain
		ID3D11Texture2D* pBuffer;
		hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer);
		AssertOrQuit(SUCCEEDED(hr));

		hr = g_pd3dDevice->CreateRenderTargetView(pBuffer, nullptr, &g_pSwapChainRTV);
		AssertOrQuit(SUCCEEDED(hr));
		pBuffer->Release();

		g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, nullptr);

		// Setup the viewport
		D3D11_VIEWPORT vp;
		vp.Width = width;
		vp.Height = height;
		vp.MinDepth = 0.0f;
		vp.MaxDepth = 1.0f;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;
		g_pd3dDeviceContext->RSSetViewports(1, &vp);


		ID3DBlob* pShader;
		ID3DBlob* pErrorMsgs;
		// Vertex shader
		{
			hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", nullptr, nullptr, "VS", "vs_4_0", 0/*Flags1*/, 0/*Flags2*/, &pShader, &pErrorMsgs);

			if (FAILED(hr))
			{
				const char* pStr = (const char*)pErrorMsgs->GetBufferPointer();
				printf(pStr);
			}

			AssertOrQuit(SUCCEEDED(hr));
			hr = g_pd3dDevice->CreateVertexShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), nullptr, &g_pVertexShader);
			AssertOrQuit(SUCCEEDED(hr));
			// Let's bind it now : no other vtx shader will replace it...
			g_pd3dDeviceContext->VSSetShader(g_pVertexShader, nullptr, 0);
			//hr = g_pd3dDevice->CreateInputLayout(...pShader used for signature...) No need
		}
		// Pixel shader
		{
			hr = D3DCompile(g_simpleShaders, strlen(g_simpleShaders), "Memory", nullptr, nullptr, "PS", "ps_4_0", 0/*Flags1*/, 0/*Flags2*/, &pShader, &pErrorMsgs);

			AssertOrQuit(SUCCEEDED(hr));
			hr = g_pd3dDevice->CreatePixelShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), nullptr, &g_pPixelShader);
			AssertOrQuit(SUCCEEDED(hr));
			// Let's bind it now : no other pix shader will replace it...
			g_pd3dDeviceContext->PSSetShader(g_pPixelShader, nullptr, 0);
		}
		// Create the constant buffer
		{
			D3D11_BUFFER_DESC cbDesc;
			cbDesc.Usage = D3D11_USAGE_DYNAMIC;
			cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;//D3D11_BIND_SHADER_RESOURCE;
			cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			cbDesc.MiscFlags = 0;
			cbDesc.ByteWidth = 16 * ((sizeof(ConstantBuffer) + 16) / 16);
			//cbDesc.StructureByteStride = 0;
			hr = g_pd3dDevice->CreateBuffer(&cbDesc, nullptr, &g_pConstantBuffer);
			AssertOrQuit(SUCCEEDED(hr));
			// Assign the buffer now : nothing in the code will interfere with this (very simple sample)
			g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
			g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
		}
		// SamplerState
		{
			D3D11_SAMPLER_DESC sDesc;
			sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			sDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
			sDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
			sDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
			sDesc.MinLOD = 0;
			sDesc.MaxLOD = 8;
			sDesc.MipLODBias = 0;
			sDesc.MaxAnisotropy = 1;
			hr = g_pd3dDevice->CreateSamplerState(&sDesc, &g_pSamplerState);
			AssertOrQuit(SUCCEEDED(hr));
			g_pd3dDeviceContext->PSSetSamplers(0, 1, &g_pSamplerState);
		}

		// Setup  no Input Layout
		g_pd3dDeviceContext->IASetInputLayout(0);
		g_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

		D3D11_RASTERIZER_DESC rasterizerState;
		rasterizerState.FillMode = D3D11_FILL_SOLID;
		rasterizerState.CullMode = D3D11_CULL_FRONT;
		rasterizerState.FrontCounterClockwise = false;
		rasterizerState.DepthBias = false;
		rasterizerState.DepthBiasClamp = 0;
		rasterizerState.SlopeScaledDepthBias = 0;
		rasterizerState.DepthClipEnable = false;
		rasterizerState.ScissorEnable = false;
		rasterizerState.MultisampleEnable = false;
		rasterizerState.AntialiasedLineEnable = false;
		g_pd3dDevice->CreateRasterizerState(&rasterizerState, &g_pRasterState);
		g_pd3dDeviceContext->RSSetState(g_pRasterState);

		return S_OK;
	}

	HRESULT InitTextures();
	void Cleanup();
	bool DrawScene();
};

