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
#include "Timer.h"


RayTracingKernel rtk;
DXManager dxm;

HWND hWnd;
WNDCLASSEX wc;
Timer timer;
float dt = 0;

bool g_bDone = false;

const unsigned int g_WindowWidth = 1280;
const unsigned int g_WindowHeight = 720;

void Cleanup()
{
	rtk.Cleanup();
	dxm.Cleanup();

	UnregisterClass(wc.lpszClassName, wc.hInstance);
}

void Render()
{
	rtk.Run();
	dxm.DrawScene();
}

void Run() {
	ShowWindow(hWnd, SW_SHOWDEFAULT);
	UpdateWindow(hWnd);
	timer.Start();

	while (false == g_bDone)
	{
		dt = timer.GetMilisecondsElapsed();
		timer.Restart();

		Render();

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
}

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

		if (rtk.sm.h_cam != nullptr) {
			Camera* cam = rtk.sm.h_cam;
			bool chagned = false;

			if (wParam == 0x51)//q
			{
				cam->position.y += dt * cam->speed;
				chagned = true;
			}
			if (wParam == 0x5A)//z
			{
				cam->position.y -= dt * cam->speed;
				chagned = true;
			}
			if (wParam == 0x57)//w
			{
				vec3 forward = cam->GetForward();
				forward.y = 0;
				cam->position += forward.norm() * cam->speed * dt;
				chagned = true;
			}
			if (wParam == 0x41)//a
			{
				vec3 right = cam->GetRight();
				right.y = 0;
				cam->position -= right.norm() * cam->speed * dt;
				chagned = true;
			}
			if (wParam == 0x53)//s
			{
				vec3 forward = cam->GetForward();
				forward.y = 0;
				cam->position -= forward.norm() * cam->speed * dt;
				chagned = true;
			}
			if (wParam == 0x44)//d
			{
				vec3 right = cam->GetRight();
				right.y = 0;
				cam->position += right.norm() * cam->speed * dt;
				chagned = true;
			}

			if (chagned) 
				rtk.sm.UpdateCamera();
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

void InitWindow() {
	wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
					  GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
					  "CUDA SDK", nullptr
	};
	RegisterClassEx(&wc);

	// Create the application's window
	int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
	int yMenu = ::GetSystemMetrics(SM_CYMENU);
	int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);
	hWnd = CreateWindow(wc.lpszClassName, "CUDA/D3D11 Texture InterOP",
		WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth + 2 * xBorder, g_WindowHeight + 2 * yBorder + yMenu,
		nullptr, nullptr, wc.hInstance, nullptr);
}


int main(int argc, char* argv[])
{
	InitWindow();

	if (!rtk.Init(g_WindowWidth, g_WindowHeight, false))
		exit(EXIT_SUCCESS);

	if (!dxm.Init(hWnd, g_WindowWidth, g_WindowHeight))
		exit(EXIT_SUCCESS);

	rtk.RegisterTexture(dxm.pTexture);


	Run();


	exit(EXIT_SUCCESS);
}