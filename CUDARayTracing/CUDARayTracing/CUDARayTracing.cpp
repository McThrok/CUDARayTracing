#pragma warning(disable: 4312)

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

#include "RayTracingKernel.h"
#include "DXManager.h"


RayTracingKernel rtk;
DXManager dxm;


// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

bool g_bDone = false;

const unsigned int g_WindowWidth = 1280;
const unsigned int g_WindowHeight = 720;


#define NAME_LEN    512

bool findCUDADevice()
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[NAME_LEN], devname[NAME_LEN];

	// This function call returns 0 if there are no CUDA capable devices.
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	// Get CUDA device properties
	cudaDeviceProp deviceProp;

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaGetDeviceProperties(&deviceProp, dev);
		STRCPY(devname, NAME_LEN, deviceProp.name);
		printf("> GPU %d: %s\n", dev, devname);
	}

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
	sd.BufferDesc.Width = dxm.width;
	sd.BufferDesc.Height = dxm.height;
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
		dxm.g_pCudaCapableAdapter,
		D3D_DRIVER_TYPE_UNKNOWN,//D3D_DRIVER_TYPE_HARDWARE,
		nullptr, //HMODULE Software
		0, //UINT Flags
		tour_fl, // D3D_FEATURE_LEVEL* pFeatureLevels
		3, //FeatureLevels
		D3D11_SDK_VERSION, //UINT SDKVersion
		&sd, // DXGI_SWAP_CHAIN_DESC* pSwapChainDesc
		&dxm.g_pSwapChain, //IDXGISwapChain** ppSwapChain
		&dxm.g_pd3dDevice, //ID3D11Device** ppDevice
		&flRes, //D3D_FEATURE_LEVEL* pFeatureLevel
		&dxm.g_pd3dDeviceContext//ID3D11DeviceContext** ppImmediateContext
	);
	AssertOrQuit(SUCCEEDED(hr));

	dxm.g_pCudaCapableAdapter->Release();

	// Get the immediate DeviceContext
	dxm.g_pd3dDevice->GetImmediateContext(&dxm.g_pd3dDeviceContext);

	// Create a render target view of the swapchain
	ID3D11Texture2D* pBuffer;
	hr = dxm.g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBuffer);
	AssertOrQuit(SUCCEEDED(hr));

	hr = dxm.g_pd3dDevice->CreateRenderTargetView(pBuffer, nullptr, &dxm.g_pSwapChainRTV);
	AssertOrQuit(SUCCEEDED(hr));
	pBuffer->Release();

	dxm.g_pd3dDeviceContext->OMSetRenderTargets(1, &dxm.g_pSwapChainRTV, nullptr);

	// Setup the viewport
	D3D11_VIEWPORT vp;
	vp.Width = dxm.width;
	vp.Height = dxm.height;
	vp.MinDepth = 0.0f;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	dxm.g_pd3dDeviceContext->RSSetViewports(1, &vp);


	ID3DBlob* pShader;
	ID3DBlob* pErrorMsgs;
	// Vertex shader
	{
		hr = D3DCompile(dxm.g_simpleShaders, strlen(dxm.g_simpleShaders), "Memory", nullptr, nullptr, "VS", "vs_4_0", 0/*Flags1*/, 0/*Flags2*/, &pShader, &pErrorMsgs);

		if (FAILED(hr))
		{
			const char* pStr = (const char*)pErrorMsgs->GetBufferPointer();
			printf(pStr);
		}

		AssertOrQuit(SUCCEEDED(hr));
		hr = dxm.g_pd3dDevice->CreateVertexShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), nullptr, &dxm.g_pVertexShader);
		AssertOrQuit(SUCCEEDED(hr));
		// Let's bind it now : no other vtx shader will replace it...
		dxm.g_pd3dDeviceContext->VSSetShader(dxm.g_pVertexShader, nullptr, 0);
		//hr = dxm.g_pd3dDevice->CreateInputLayout(...pShader used for signature...) No need
	}
	// Pixel shader
	{
		hr = D3DCompile(dxm.g_simpleShaders, strlen(dxm.g_simpleShaders), "Memory", nullptr, nullptr, "PS", "ps_4_0", 0/*Flags1*/, 0/*Flags2*/, &pShader, &pErrorMsgs);

		AssertOrQuit(SUCCEEDED(hr));
		hr = dxm.g_pd3dDevice->CreatePixelShader(pShader->GetBufferPointer(), pShader->GetBufferSize(), nullptr, &dxm.g_pPixelShader);
		AssertOrQuit(SUCCEEDED(hr));
		// Let's bind it now : no other pix shader will replace it...
		dxm.g_pd3dDeviceContext->PSSetShader(dxm.g_pPixelShader, nullptr, 0);
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
		hr = dxm.g_pd3dDevice->CreateBuffer(&cbDesc, nullptr, &dxm.g_pConstantBuffer);
		AssertOrQuit(SUCCEEDED(hr));
		// Assign the buffer now : nothing in the code will interfere with this (very simple sample)
		dxm.g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &dxm.g_pConstantBuffer);
		dxm.g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &dxm.g_pConstantBuffer);
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
		hr = dxm.g_pd3dDevice->CreateSamplerState(&sDesc, &dxm.g_pSamplerState);
		AssertOrQuit(SUCCEEDED(hr));
		dxm.g_pd3dDeviceContext->PSSetSamplers(0, 1, &dxm.g_pSamplerState);
	}

	// Setup  no Input Layout
	dxm.g_pd3dDeviceContext->IASetInputLayout(0);
	dxm.g_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

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
	dxm.g_pd3dDevice->CreateRasterizerState(&rasterizerState, &dxm.g_pRasterState);
	dxm.g_pd3dDeviceContext->RSSetState(dxm.g_pRasterState);

	return S_OK;
}

HRESULT InitTextures()
{
	//
	// create the D3D resources we'll be using
	//
	// 2D texture
	{
		rtk.width = g_WindowWidth;
		rtk.height = g_WindowHeight;

		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
		desc.Width = rtk.width;
		desc.Height = rtk.height;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		desc.SampleDesc.Count = 1;
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

		if (FAILED(dxm.g_pd3dDevice->CreateTexture2D(&desc, nullptr, &dxm.pTexture)))
		{
			return E_FAIL;
		}

		if (FAILED(dxm.g_pd3dDevice->CreateShaderResourceView(dxm.pTexture, nullptr, &dxm.pSRView)))
		{
			return E_FAIL;
		}

		dxm.offsetInShader = 0; // to be clean we should look for the offset from the shader code
		dxm.g_pd3dDeviceContext->PSSetShaderResources(dxm.offsetInShader, 1, &dxm.pSRView);
	}

	return S_OK;
}



void Cleanup()
{
	//
	// clean up Direct3D
	//
	rtk.Cleanup();

	{

		dxm.Cleanup();
	}
}

void Render()
{
	rtk.Run();

	dxm.DrawScene();
}

static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_KEYDOWN:
		if (wParam == VK_ESCAPE)
		{
			g_bDone = true;
			Cleanup();
			PostQuitMessage(0);
			return 0;
		}

		break;

	case WM_DESTROY:
		g_bDone = true;
		Cleanup();
		PostQuitMessage(0);
		return 0;

	case WM_PAINT:
		ValidateRect(hWnd, nullptr);
		return 0;
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}

int main(int argc, char* argv[])
{
	char device_name[256];

	if (!findCUDADevice())                   // Search for CUDA GPU
	{
		printf("> CUDA Device NOT found on \"%s\".. Exiting.\n", device_name);
		exit(EXIT_SUCCESS);
	}

	if (!dynlinkLoadD3D11API())                  // Search for D3D API (locate drivers, does not mean device is found)
	{
		printf("> D3D11 API libraries NOT found on.. Exiting.\n");
		dynlinkUnloadD3D11API();
		exit(EXIT_SUCCESS);
	}

	if (!dxm.findDXDevice(device_name))           // Search for D3D Hardware Device
	{
		printf("> D3D11 Graphics Device NOT found.. Exiting.\n");
		dynlinkUnloadD3D11API();
		exit(EXIT_SUCCESS);
	}


	//
	// create window
	//
	// Register the window class
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
					  GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
					  "CUDA SDK", nullptr
	};
	RegisterClassEx(&wc);

	// Create the application's window
	int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
	int yMenu = ::GetSystemMetrics(SM_CYMENU);
	int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
	HWND hWnd = CreateWindow(wc.lpszClassName, "CUDA/D3D11 Texture InterOP",
		WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu,
		nullptr, nullptr, wc.hInstance, nullptr);


	ShowWindow(hWnd, SW_SHOWDEFAULT);
	UpdateWindow(hWnd);

	dxm.width = g_WindowWidth;
	dxm.height = g_WindowHeight;

	// Initialize Direct3D
	if (SUCCEEDED(InitD3D(hWnd)) &&
		SUCCEEDED(InitTextures()))
	{
		rtk.InitSpheres();
		// 2D
		// register the Direct3D resources that we'll use
		// we'll read to and write from g_texture_2d, so don't set any special map flags for it
		cudaGraphicsD3D11RegisterResource(&rtk.cudaResource, dxm.pTexture, cudaGraphicsRegisterFlagsNone);
		getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
		// cuda cannot write into the texture directly : the texture is seen as a cudaArray and can only be mapped as a texture
		// Create a buffer so that cuda can write into it
		// pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
		cudaMallocPitch(&rtk.cudaLinearMemory, &rtk.pitch, rtk.width * sizeof(float) * 4, rtk.height);
		getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
		cudaMemset(rtk.cudaLinearMemory, 1, rtk.pitch * rtk.height);
	}

	//
	// the main loop
	//
	while (false == g_bDone)
	{
		Render();

		//
		// handle I/O
		//
		MSG msg;
		ZeroMemory(&msg, sizeof(msg));

		while (msg.message != WM_QUIT)
		{
			if (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
			else
			{
				Render();
			}
		}

	};

	// Release D3D Library (after message loop)
	dynlinkUnloadD3D11API();

	// Unregister windows class
	UnregisterClass(wc.lpszClassName, wc.hInstance);


	exit(EXIT_SUCCESS);
}