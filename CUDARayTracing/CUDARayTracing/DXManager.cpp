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


bool DXManager::loadDevice() {

	// Search for D3D Hardware Device

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

	char device_name[256];
	wcstombs(device_name, adapterDesc.Description, 128);

	printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n");
	printf("> %s\n", device_name);

	return true;
}

bool DXManager::findDXDevice()
{
	// Search for D3D API (locate drivers, does not mean device is found)
	if (!dynlinkLoadD3D11API())
		return false;

	if (loadDevice())
		return true;

	dynlinkUnloadD3D11API();
	return false;
}

HRESULT DXManager::InitD3D(HWND hWnd)
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

HRESULT DXManager::InitTextures()
{
	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = width;
	desc.Height = height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	if (FAILED(g_pd3dDevice->CreateTexture2D(&desc, nullptr, &pTexture)))
	{
		return E_FAIL;
	}

	if (FAILED(g_pd3dDevice->CreateShaderResourceView(pTexture, nullptr, &pSRView)))
	{
		return E_FAIL;
	}

	offsetInShader = 0; // to be clean we should look for the offset from the shader code
	g_pd3dDeviceContext->PSSetShaderResources(offsetInShader, 1, &pSRView);

	return S_OK;
}

void DXManager::Cleanup()
{
	// release the resources we created
	pSRView->Release();
	pTexture->Release();

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

	// Release D3D Library (after message loop)
	dynlinkUnloadD3D11API();
}