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

bool g_bDone = false;

const unsigned int g_WindowWidth = 1280;
const unsigned int g_WindowHeight = 720;

#define NAME_LEN    512

void Cleanup()
{
	rtk.Cleanup();
	dxm.Cleanup();
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

	if (!rtk.findCUDADevice())                   // Search for CUDA GPU
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
	rtk.width = g_WindowWidth;
	rtk.height = g_WindowHeight;

	// Initialize Direct3D
	if (SUCCEEDED(dxm.InitD3D(hWnd)) && SUCCEEDED(dxm.InitTextures()))
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