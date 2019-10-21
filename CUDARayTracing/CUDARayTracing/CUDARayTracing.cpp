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


//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
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

// Data structure for 2D texture shared between DX10 and CUDA
struct
{
	ID3D11Texture2D* pTexture;
	ID3D11ShaderResourceView* pSRView;
	int offsetInShader;

} g_texture_2d;

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

	for (; !dxm.g_pCudaCapableAdapter; ++adapter)
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
			dxm.g_pCudaCapableAdapter = pAdapter;
			dxm.g_pCudaCapableAdapter->AddRef();
		}

		pAdapter->Release();
	}

	printf("> Found %d D3D11 Adapater(s).\n", (int)adapter);

	pFactory->Release();

	if (!dxm.g_pCudaCapableAdapter)
	{
		printf("> Found 0 D3D11 Adapater(s) /w Compute capability.\n");
		return false;
	}

	DXGI_ADAPTER_DESC adapterDesc;
	dxm.g_pCudaCapableAdapter->GetDesc(&adapterDesc);
	wcstombs(dev_name, adapterDesc.Description, 128);

	printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
	printf("> %s\n", dev_name);

	return true;
}

//-----------------------------------------------------------------------------
// Name: InitD3D()
// Desc: Initializes Direct3D
//-----------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd)
{
	HRESULT hr = S_OK;
	cudaError cuStatus;

	// Set up the structure used to create the device and swapchain
	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory(&sd, sizeof(sd));
	sd.BufferCount = 1;
	sd.BufferDesc.Width = g_WindowWidth;
	sd.BufferDesc.Height = g_WindowHeight;
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
	vp.Width = g_WindowWidth;
	vp.Height = g_WindowHeight;
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

//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
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

		if (FAILED(dxm.g_pd3dDevice->CreateTexture2D(&desc, nullptr, &g_texture_2d.pTexture)))
		{
			return E_FAIL;
		}

		if (FAILED(dxm.g_pd3dDevice->CreateShaderResourceView(g_texture_2d.pTexture, nullptr, &g_texture_2d.pSRView)))
		{
			return E_FAIL;
		}

		g_texture_2d.offsetInShader = 0; // to be clean we should look for the offset from the shader code
		dxm.g_pd3dDeviceContext->PSSetShaderResources(g_texture_2d.offsetInShader, 1, &g_texture_2d.pSRView);
	}

	return S_OK;
}

void InitSpheres() {
	rtk.spheres_num = 1;

	unsigned int mem_size = sizeof(float) * 4 * rtk.spheres_num;
	float* h_spheres = (float*)malloc(mem_size);

	h_spheres[0] = 0;
	h_spheres[1] = 0;
	h_spheres[2] = 0;
	h_spheres[3] = 0.5f;

	checkCudaErrors(cudaMalloc((void**)&rtk.spheres, mem_size));
	checkCudaErrors(cudaMemcpy(rtk.spheres, h_spheres, mem_size, cudaMemcpyHostToDevice));

	free(h_spheres);
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
bool DrawScene()
{
	// Clear the backbuffer to a black color
	float ClearColor[4] = { 0.5f, 0.5f, 0.6f, 1.0f };
	dxm.g_pd3dDeviceContext->ClearRenderTargetView(dxm.g_pSwapChainRTV, ClearColor);

	float quadRect[4] = { -1,  -1,  2 , 2 };
	//
	// draw the 2d texture
	//
	HRESULT hr;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	ConstantBuffer* pcb;
	hr = dxm.g_pd3dDeviceContext->Map(dxm.g_pConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	AssertOrQuit(SUCCEEDED(hr));
	pcb = (ConstantBuffer*)mappedResource.pData;
	{
		memcpy(pcb->vQuadRect, quadRect, sizeof(float) * 4);
	}
	dxm.g_pd3dDeviceContext->Unmap(dxm.g_pConstantBuffer, 0);
	dxm.g_pd3dDeviceContext->Draw(4, 0);

	// Present the backbuffer contents to the display
	dxm.g_pSwapChain->Present(0, 0);
	return true;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup()
{
	//
	// clean up Direct3D
	//
	rtk.Cleanup();

	{
		// release the resources we created
		g_texture_2d.pSRView->Release();
		g_texture_2d.pTexture->Release();


		if (dxm.g_pVertexShader)
			dxm.g_pVertexShader->Release();

		if (dxm.g_pPixelShader)
			dxm.g_pPixelShader->Release();

		if (dxm.g_pConstantBuffer)
			dxm.g_pConstantBuffer->Release();

		if (dxm.g_pSamplerState)
			dxm.g_pSamplerState->Release();

		if (dxm.g_pSwapChainRTV != nullptr)
			dxm.g_pSwapChainRTV->Release();

		if (dxm.g_pSwapChain != nullptr)
			dxm.g_pSwapChain->Release();

		if (dxm.g_pd3dDevice != nullptr)
			dxm.g_pd3dDevice->Release();
	}
}

//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void Render()
{
	rtk.Run();

	//
	// draw the scene using them
	//
	DrawScene();
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
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

	if (!findDXDevice(device_name))           // Search for D3D Hardware Device
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

	// Initialize Direct3D
	if (SUCCEEDED(InitD3D(hWnd)) &&
		SUCCEEDED(InitTextures()))
	{
		InitSpheres();
		// 2D
		// register the Direct3D resources that we'll use
		// we'll read to and write from g_texture_2d, so don't set any special map flags for it
		cudaGraphicsD3D11RegisterResource(&rtk.cudaResource, g_texture_2d.pTexture, cudaGraphicsRegisterFlagsNone);
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