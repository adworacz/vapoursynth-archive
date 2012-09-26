#define FP_STATE 0x9001f

#define INITGUID
#define WIN32_LEAN_AND_MEAN
#include <objbase.h>
#include <vfw.h>
#include <windows.h>
#include <cstdio>
#include <cassert>
#include <string>

// fixme
#include "../../include/VapourSynth.h"
#include "../../src/cython/vapoursynthpp_api.h"

const VSAPI *vsapi;

#define UNDEFINED_LENGTH 10000000

void BitBlt(uint8_t* dstp, int dst_pitch, const uint8_t* srcp, int src_pitch, int row_size, int height) {
	for (int i = 0; i < height; i++) {
		memcpy(dstp, srcp, row_size);
		srcp += src_pitch;
		dstp += dst_pitch;
	}
}

static long refCount=0;

static int BMPSize(int height, int rowsize, int planar) {
	if (planar) {
		int p = height * ((rowsize+3) & ~3);
		p+=p>>1;
		return p; 
	} else
		return height * ((rowsize+3) & ~3);
}

static int ImageSize(const VSVideoInfo *vi) {
    if (!vi)
        return 0;
	  int image_size;
	
      if (vi->format->numPlanes == 1) {
          image_size = BMPSize(vi->height, vi->width * vi->format->bytesPerSample, 0);
	  } else { // Packed size
        image_size = (vi->width * vi->format->bytesPerSample) >> vi->format->subSamplingW;
	    if (image_size) {
	      image_size  *= vi->height;
          image_size >>= vi->format->subSamplingH;
	      image_size  *= 2;
	    }
        image_size += vi->width * vi->format->bytesPerSample * vi->height;
	  }
	  return image_size;
	}

// {58F74CA0-BD0E-4664-A49B-8D10E6F0C131}
extern "C" const GUID CLSID_VapourSynth = 
{ 0x58f74ca0, 0xbd0e, 0x4664, { 0xa4, 0x9b, 0x8d, 0x10, 0xe6, 0xf0, 0xc1, 0x31 } };

extern "C" const GUID IID_IAvisynthClipInfo   // {E6D6B708-124D-11D4-86F3-DB80AFD98778}
  = {0xe6d6b708, 0x124d, 0x11d4, {0x86, 0xf3, 0xdb, 0x80, 0xaf, 0xd9, 0x87, 0x78}};

struct IAvisynthClipInfo : IUnknown {
  virtual int __stdcall GetError(const char** ppszMessage) = 0;
  virtual bool __stdcall GetParity(int n) = 0;
  virtual bool __stdcall IsFieldBased() = 0;
};

class VapourSynthFile: public IAVIFile, public IPersistFile, public IClassFactory, public IAvisynthClipInfo {
	friend class VapourSynthStream;
private:
	long m_refs;

    std::string szScriptName;
	ScriptExport se;
    const VSVideoInfo* vi;
    const char* error_msg;

    CRITICAL_SECTION cs_filter_graph;

    bool DelayInit();
    bool DelayInit2();

    void MakeErrorStream(const char* msg);

    void Lock();
    void Unlock();

public:

	VapourSynthFile(const CLSID& rclsid);
	~VapourSynthFile();

	static HRESULT Create(const CLSID& rclsid, const IID& riid, void **ppv);

	//////////// IUnknown

	STDMETHODIMP QueryInterface(const IID& iid, void **ppv);
	STDMETHODIMP_(ULONG) AddRef();
	STDMETHODIMP_(ULONG) Release();

	//////////// IClassFactory

	STDMETHODIMP CreateInstance (LPUNKNOWN pUnkOuter, REFIID riid,  void * * ppvObj) ;
	STDMETHODIMP LockServer (BOOL fLock) ;

	//////////// IPersistFile

	STDMETHODIMP GetClassID(LPCLSID lpClassID);  // IPersist

	STDMETHODIMP IsDirty();
	STDMETHODIMP Load(LPCOLESTR lpszFileName, DWORD grfMode);
	STDMETHODIMP Save(LPCOLESTR lpszFileName, BOOL fRemember);
	STDMETHODIMP SaveCompleted(LPCOLESTR lpszFileName);
	STDMETHODIMP GetCurFile(LPOLESTR *lplpszFileName);

	//////////// IAVIFile

	STDMETHODIMP CreateStream(PAVISTREAM *ppStream, AVISTREAMINFOW *psi);       // 5
	STDMETHODIMP EndRecord();                                                   // 8
	STDMETHODIMP GetStream(PAVISTREAM *ppStream, DWORD fccType, LONG lParam);   // 4
	STDMETHODIMP Info(AVIFILEINFOW *psi, LONG lSize);                           // 3

	STDMETHODIMP Open(LPCSTR szFile, UINT mode, LPCOLESTR lpszFileName);        // ???
    STDMETHODIMP Save(LPCSTR szFile, AVICOMPRESSOPTIONS FAR *lpOptions,         // ???
				AVISAVECALLBACK lpfnCallback);

	STDMETHODIMP ReadData(DWORD fcc, LPVOID lp, LONG *lpcb);                    // 7
	STDMETHODIMP WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer);          // 6
	STDMETHODIMP DeleteStream(DWORD fccType, LONG lParam);                      // 9

	//////////// IAvisynthClipInfo

	int __stdcall GetError(const char** ppszMessage);
	bool __stdcall GetParity(int n);
	bool __stdcall IsFieldBased();
};

///////////////////////////////////

class VapourSynthStream: public IAVIStream , public IAVIStreaming {
public:

	//////////// IUnknown

	STDMETHODIMP QueryInterface(const IID& iid, void **ppv);
	STDMETHODIMP_(ULONG) AddRef();
	STDMETHODIMP_(ULONG) Release();

	VapourSynthStream(VapourSynthFile *parentPtr, bool isAudio);
	~VapourSynthStream();

	//////////// IAVIStream

	STDMETHODIMP Create(LONG lParam1, LONG lParam2);
	STDMETHODIMP Delete(LONG lStart, LONG lSamples);
	STDMETHODIMP_(LONG) Info(AVISTREAMINFOW *psi, LONG lSize);
	STDMETHODIMP_(LONG) FindSample(LONG lPos, LONG lFlags);
	STDMETHODIMP Read(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples);
	STDMETHODIMP ReadData(DWORD fcc, LPVOID lp, LONG *lpcb);
	STDMETHODIMP ReadFormat(LONG lPos, LPVOID lpFormat, LONG *lpcbFormat);
	STDMETHODIMP SetFormat(LONG lPos, LPVOID lpFormat, LONG cbFormat);
	STDMETHODIMP Write(LONG lStart, LONG lSamples, LPVOID lpBuffer,
		LONG cbBuffer, DWORD dwFlags, LONG FAR *plSampWritten, 
		LONG FAR *plBytesWritten);
	STDMETHODIMP WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer);
	STDMETHODIMP SetInfo(AVISTREAMINFOW *psi, LONG lSize);

	//////////// IAVIStreaming
	
	STDMETHODIMP Begin(LONG lStart, LONG lEnd, LONG lRate);
	STDMETHODIMP End();

private:
	long m_refs;

	VapourSynthFile *parent;
	std::string sName;

	//////////// internal

	void ReadHelper(void* lpBuffer, int lStart, int lSamples, unsigned code[4]);
	void ReadFrame(void* lpBuffer, int n);

	HRESULT Read2(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples);
};


BOOL APIENTRY DllMain(HANDLE hModule, ULONG ulReason, LPVOID lpReserved) {
	if (ulReason == DLL_PROCESS_ATTACH) {
		vsapi = getVapourSynthAPI(VAPOURSYNTH_API_VERSION);
		Py_Initialize();
		import_vapoursynth();
	} else if (ulReason == DLL_PROCESS_DETACH) {
		Py_Finalize();
	}
	return TRUE;
}

// From the Microsoft AVIFile docs.  Dense code...

extern "C" STDAPI DllGetClassObject(const CLSID& rclsid, const IID& riid, void **ppv);

STDAPI DllGetClassObject(const CLSID& rclsid, const IID& riid, void **ppv) {

	if (rclsid != CLSID_VapourSynth)
        return CLASS_E_CLASSNOTAVAILABLE;
	HRESULT hresult = VapourSynthFile::Create(rclsid, riid, ppv);
	return hresult;
}

extern "C" STDAPI DllCanUnloadNow();

STDAPI DllCanUnloadNow() {
	return refCount ? S_FALSE : S_OK;
}


///////////////////////////////////////////////////////////////////////////
//
//	VapourSynthFile
//
///////////////////////////////////////////////////////////////////////////
//////////// IClassFactory

STDMETHODIMP VapourSynthFile::CreateInstance (LPUNKNOWN pUnkOuter, REFIID riid,  void * * ppvObj) {
	if (pUnkOuter) 
      return CLASS_E_NOAGGREGATION;
	HRESULT hresult = Create(CLSID_VapourSynth, riid, ppvObj);
	return hresult;
}

STDMETHODIMP VapourSynthFile::LockServer (BOOL fLock) {
	return S_OK;
}

///////////////////////////////////////////////////
//////////// IPersistFile

STDMETHODIMP VapourSynthFile::GetClassID(LPCLSID lpClassID) {  // IPersist
    if (!lpClassID)
		return E_POINTER;
    *lpClassID = CLSID_VapourSynth;
	return S_OK;
}

STDMETHODIMP VapourSynthFile::IsDirty() {
	return S_FALSE;
}

STDMETHODIMP VapourSynthFile::Load(LPCOLESTR lpszFileName, DWORD grfMode) {
	char filename[MAX_PATH];
	WideCharToMultiByte(AreFileApisANSI() ? CP_ACP : CP_OEMCP, 0, lpszFileName, -1, filename, sizeof filename, NULL, NULL); 
	return Open(filename, grfMode, lpszFileName);
}

STDMETHODIMP VapourSynthFile::Save(LPCOLESTR lpszFileName, BOOL fRemember) {
	return E_FAIL;
}

STDMETHODIMP VapourSynthFile::SaveCompleted(LPCOLESTR lpszFileName) {
	return S_OK;
}

STDMETHODIMP VapourSynthFile::GetCurFile(LPOLESTR *lplpszFileName) {
	if (lplpszFileName)
		*lplpszFileName = NULL;
	return E_FAIL;
}

///////////////////////////////////////////////////
/////// static local

HRESULT VapourSynthFile::Create(const CLSID& rclsid, const IID& riid, void **ppv) {
	HRESULT hresult;
	VapourSynthFile* pAVIFileSynth = new VapourSynthFile(rclsid);
	if (!pAVIFileSynth)
		return E_OUTOFMEMORY;
	hresult = pAVIFileSynth->QueryInterface(riid, ppv);
	pAVIFileSynth->Release();
	return hresult;
}

///////////////////////////////////////////////////
//////////// IUnknown

STDMETHODIMP VapourSynthFile::QueryInterface(const IID& iid, void **ppv) {
	if (!ppv)
        return E_POINTER;

	if (iid == IID_IUnknown) {
		*ppv = (IUnknown *)(IAVIFile *)this;
	} else if (iid == IID_IClassFactory) {
		*ppv = (IClassFactory *)this;
	} else if (iid == IID_IPersist) {
		*ppv = (IPersist *)this;
	} else if (iid == IID_IPersistFile) {
		*ppv = (IPersistFile *)this;
	} else if (iid == IID_IAVIFile) {
		*ppv = (IAVIFile *)this;
	} else if (iid == IID_IAvisynthClipInfo) {
		*ppv = (IAvisynthClipInfo *)this;
	} else {
		*ppv = NULL;
		return E_NOINTERFACE;
	}

	AddRef();

	return S_OK;
}

STDMETHODIMP_(ULONG) VapourSynthFile::AddRef() {
	const int refs = InterlockedIncrement(&m_refs);
    InterlockedIncrement(&refCount);
	return refs;
}

STDMETHODIMP_(ULONG) VapourSynthFile::Release() {
    const int refs = InterlockedDecrement(&m_refs);
	InterlockedDecrement(&refCount);
	if (!refs)
		delete this;
	return refs;
}

////////////////////////////////////////////////////////////////////////
//
//		VapourSynthStream
//
////////////////////////////////////////////////////////////////////////
//////////// IUnknown


STDMETHODIMP VapourSynthStream::QueryInterface(const IID& iid, void **ppv) {
	if (!ppv)
        return E_POINTER;

	if (iid == IID_IUnknown) {
		*ppv = (IUnknown *)(IAVIStream *)this;
	} else if (iid == IID_IAVIStream) {
		*ppv = (IAVIStream *)this;
	} else if (iid == IID_IAVIStreaming) {
		*ppv = (IAVIStreaming *)this;
	} else {
		*ppv = NULL;
		return E_NOINTERFACE;
	}

	AddRef();

	return S_OK;
}

STDMETHODIMP_(ULONG) VapourSynthStream::AddRef() {
	const int refs = InterlockedIncrement(&m_refs);
    InterlockedIncrement(&refCount);
	return refs;
}

STDMETHODIMP_(ULONG) VapourSynthStream::Release() {
    const int refs = InterlockedDecrement(&m_refs);
    InterlockedDecrement(&refCount);
	if (!refs) delete this;
	return refs;
}

////////////////////////////////////////////////////////////////////////
//
//		VapourSynthFile
//
////////////////////////////////////////////////////////////////////////
//////////// IAVIFile

STDMETHODIMP VapourSynthFile::CreateStream(PAVISTREAM *ppStream, AVISTREAMINFOW *psi) {
	*ppStream = NULL;
	return S_OK;
}

STDMETHODIMP VapourSynthFile::EndRecord() {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthFile::Save(LPCSTR szFile, AVICOMPRESSOPTIONS FAR *lpOptions,
				AVISAVECALLBACK lpfnCallback) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthFile::ReadData(DWORD fcc, LPVOID lp, LONG *lpcb) {
	return AVIERR_NODATA;
}

STDMETHODIMP VapourSynthFile::WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthFile::DeleteStream(DWORD fccType, LONG lParam) {
	return AVIERR_READONLY;
}


///////////////////////////////////////////////////
/////// local

VapourSynthFile::VapourSynthFile(const CLSID& rclsid) {
	m_refs = 0;
    AddRef();
    error_msg = 0;
    vi = NULL;
    InitializeCriticalSection(&cs_filter_graph);
}

VapourSynthFile::~VapourSynthFile() {
    Lock();
    if (vi) {
        vi = NULL;
        free_script(&se);
    }
    DeleteCriticalSection(&cs_filter_graph);
}


STDMETHODIMP VapourSynthFile::Open(LPCSTR szFile, UINT mode, LPCOLESTR lpszFileName) {
    if (mode & (OF_CREATE|OF_WRITE))
      return E_FAIL;
    szScriptName = szFile;
    return S_OK;
}

bool VapourSynthFile::DelayInit() {
    Lock();
    bool result = DelayInit2();
    Unlock();
    return result;
}

bool VapourSynthFile::DelayInit2() {
	 if (!szScriptName.empty() && !vi) {
		try {
			if (!evaluate_script((char *)szScriptName.c_str(), &se)) {
				vi = vsapi->getVideoInfo(se.node);
			} else {
				vi = NULL;
			}
		} catch (...) {
			return false;
		}
		return true;

	} else {
		return !!vi;
	}
}


void VapourSynthFile::MakeErrorStream(const char* msg) {
  error_msg = msg;
  _ASSERTE(false);
  //filter_graph = Create_MessageClip(msg, vi->width, vi->height, vi->pixel_type, false, 0xFF3333, 0, 0, env);
}

void VapourSynthFile::Lock() {
  EnterCriticalSection(&cs_filter_graph);
}

void VapourSynthFile::Unlock() {
  LeaveCriticalSection(&cs_filter_graph);
}

///////////////////////////////////////////////////
//////////// IAVIFile

STDMETHODIMP VapourSynthFile::Info(AVIFILEINFOW *pfi, LONG lSize) {
	if (!pfi)
		return E_POINTER;

	if (!DelayInit())
		return E_FAIL;

	AVIFILEINFOW afi;
    memset(&afi, 0, sizeof(afi));

	afi.dwMaxBytesPerSec	= 0;
	afi.dwFlags				= AVIFILEINFO_HASINDEX | AVIFILEINFO_ISINTERLEAVED;
	afi.dwCaps				= AVIFILECAPS_CANREAD | AVIFILECAPS_ALLKEYFRAMES | AVIFILECAPS_NOCOMPRESSION;

	afi.dwStreams				= 1;
	afi.dwSuggestedBufferSize	= 0;
	afi.dwWidth					= vi->width;
	afi.dwHeight				= vi->height;
	afi.dwEditCount				= 0;

	afi.dwRate					= vi->fpsNum ? vi->fpsNum : 1;
	afi.dwScale					= vi->fpsDen ? vi->fpsDen : 30;
	afi.dwLength				= vi->numFrames ? vi->numFrames : UNDEFINED_LENGTH;

	wcscpy(afi.szFileType, L"VapourSynth");

// Maybe should return AVIERR_BUFFERTOOSMALL for lSize < sizeof(afi)
    memset(pfi, 0, lSize);
    memcpy(pfi, &afi, min(size_t(lSize), sizeof(afi)));
	return S_OK;
}

static inline char BePrintable(int ch) {
  ch &= 0xff;
  return isprint(ch) ? ch : '.';
}


STDMETHODIMP VapourSynthFile::GetStream(PAVISTREAM *ppStream, DWORD fccType, LONG lParam) {
	VapourSynthStream *casr;
	char fcc[5];

	fcc[0] = BePrintable(fccType      );
	fcc[1] = BePrintable(fccType >>  8);
	fcc[2] = BePrintable(fccType >> 16);
	fcc[3] = BePrintable(fccType >> 24);
	fcc[4] = 0;

	if (!DelayInit())
		return E_FAIL;

    *ppStream = NULL;

	if (!fccType) {
		if (lParam==0)
			fccType = streamtypeVIDEO;
	}

	if (lParam > 0)
		return AVIERR_NODATA;

	if (fccType == streamtypeVIDEO) {
        if ((casr = new VapourSynthStream(this, false)) == 0)
			return AVIERR_MEMORY;

		*ppStream = (IAVIStream *)casr;

	} else if (fccType == streamtypeAUDIO) {
		return AVIERR_NODATA;

		if ((casr = new VapourSynthStream(this, true)) == 0)
			return AVIERR_MEMORY;
		*ppStream = (IAVIStream *)casr;
	} else
		return AVIERR_NODATA;

	return S_OK;
}


////////////////////////////////////////////////////////////////////////
//////////// IAvisynthClipInfo

int __stdcall VapourSynthFile::GetError(const char** ppszMessage) {
  if (!DelayInit() && !error_msg)
    error_msg = "VapourSynth: script open failed!";

  if (ppszMessage)
    *ppszMessage = error_msg;
  return !!error_msg;
}

bool __stdcall VapourSynthFile::GetParity(int n) {
  if (!DelayInit())
    return false;
  return false;
}

bool __stdcall VapourSynthFile::IsFieldBased() {
  if (!DelayInit())
    return false;
  return false;
}

////////////////////////////////////////////////////////////////////////
//
//		VapourSynthStream
//
////////////////////////////////////////////////////////////////////////
//////////// IAVIStreaming

STDMETHODIMP VapourSynthStream::Begin(LONG lStart, LONG lEnd, LONG lRate) {
	return S_OK;
}

STDMETHODIMP VapourSynthStream::End() {
	return S_OK;
}

//////////// IAVIStream

STDMETHODIMP VapourSynthStream::Create(LONG lParam1, LONG lParam2) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthStream::Delete(LONG lStart, LONG lSamples) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthStream::ReadData(DWORD fcc, LPVOID lp, LONG *lpcb) {
	return AVIERR_NODATA;
}

STDMETHODIMP VapourSynthStream::SetFormat(LONG lPos, LPVOID lpFormat, LONG cbFormat) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthStream::WriteData(DWORD fcc, LPVOID lpBuffer, LONG cbBuffer) {
	return AVIERR_READONLY;
}

STDMETHODIMP VapourSynthStream::SetInfo(AVISTREAMINFOW *psi, LONG lSize) {
	return AVIERR_READONLY;
}

////////////////////////////////////////////////////////////////////////
//////////// local

VapourSynthStream::VapourSynthStream(VapourSynthFile *parentPtr, bool isAudio) {
  sName = "video";
  m_refs = 0;
  AddRef();
  parent = parentPtr;
  parent->AddRef();
}

VapourSynthStream::~VapourSynthStream() {
  if (parent)
    parent->Release();
}

////////////////////////////////////////////////////////////////////////
//////////// IAVIStream

STDMETHODIMP_(LONG) VapourSynthStream::Info(AVISTREAMINFOW *psi, LONG lSize) {
	if (!psi)
		return E_POINTER;

	AVISTREAMINFOW asi;

    const VSVideoInfo* const vi = parent->vi;

    memset(&asi, 0, sizeof(asi));
    asi.fccType = streamtypeVIDEO;
    asi.dwQuality = DWORD(-1);

    const int image_size = ImageSize(vi);
      asi.fccHandler = 'UNKN';
      if (vi->format->id == pfCompatBGR32)
        asi.fccHandler = ' BID';
      else if (vi->format->id == pfCompatYUY2)
        asi.fccHandler = '2YUY';
      else if (vi->format->id == pfYUV420P8)
		asi.fccHandler = '21VY'; 
      else if (vi->format->id == pfGray8)
        asi.fccHandler = '008Y'; 
      else if (vi->format->id == pfYUV444P8)
        asi.fccHandler = '42VY'; 
      else if (vi->format->id == pfYUV420P8)
        asi.fccHandler = '61VY'; 
      else if (vi->format->id == pfYUV411P8)
        asi.fccHandler = 'B14Y'; 
      else if (vi->format->id == pfYUV410P8) 
	    asi.fccHandler = '9UVY'; 
      else {
        _ASSERT(FALSE);
      }

	  asi.dwScale = vi->fpsDen ? vi->fpsDen : 1;
      asi.dwRate = vi->fpsNum ? vi->fpsNum : 30;
	  asi.dwLength = vi->numFrames ? vi->numFrames : UNDEFINED_LENGTH;
      asi.rcFrame.right = vi->width;
      asi.rcFrame.bottom = vi->height;
      asi.dwSampleSize = image_size;
      asi.dwSuggestedBufferSize = image_size;
      wcscpy(asi.szName, L"VapourSynth video #1");

// Maybe should return AVIERR_BUFFERTOOSMALL for lSize < sizeof(asi)
    memset(psi, 0, lSize);
    memcpy(psi, &asi, min(size_t(lSize), sizeof(asi)));
	return S_OK;
}

STDMETHODIMP_(LONG) VapourSynthStream::FindSample(LONG lPos, LONG lFlags) {
	if (lFlags & FIND_FORMAT)
		return -1;

	if (lFlags & FIND_FROM_START)
		return 0;

	return lPos;
}


////////////////////////////////////////////////////////////////////////
//////////// local

void VapourSynthStream::ReadFrame(void* lpBuffer, int n) {
	const VSFrameRef *f = vsapi->getFrame(n, parent->se.node, 0, 0);
	if (!f)
		_ASSERTE(false);
    //parent->env->ThrowError("Avisynth error: generated video frame was nil (this is a bug)");

    const VSFormat *fi = vsapi->getFrameFormat(f);
	const int pitch    = vsapi->getStride(f, 0);
  const int row_size = vsapi->getFrameWidth(f, 0) * fi->bytesPerSample;
  const int height   = vsapi->getFrameHeight(f, 0);

    int out_pitch;
	  int out_pitchUV;
	
	  // BMP scanlines are dword-aligned
      if (fi->numPlanes == 1) {
	    out_pitch = (row_size+3) & ~3;
        out_pitchUV = (vsapi->getFrameWidth(f, 1) * fi->bytesPerSample+3) & ~3;
	  }
	  // Planar scanlines are packed
	  else {
	    out_pitch = row_size;
	    out_pitchUV = vsapi->getFrameWidth(f, 1) * fi->bytesPerSample;
	  }

  BitBlt((BYTE*)lpBuffer, out_pitch, vsapi->getReadPtr(f, 0), pitch, row_size, height);

  if (fi->numPlanes == 3) {
      BitBlt((BYTE*)lpBuffer + (out_pitch*height),
             out_pitchUV,               vsapi->getReadPtr(f, 2),
		     vsapi->getStride(f, 2), vsapi->getFrameWidth(f, 2),
		     vsapi->getFrameHeight(f, 2) );

      BitBlt((BYTE*)lpBuffer + (out_pitch*height + vsapi->getFrameHeight(f, 1)*out_pitchUV),
             out_pitchUV,               vsapi->getReadPtr(f, 1),
		     vsapi->getStride(f, 1), vsapi->getFrameWidth(f, 1),
		     vsapi->getFrameHeight(f, 1) );
  }
  vsapi->freeFrame(f);
}

void VapourSynthStream::ReadHelper(void* lpBuffer, int lStart, int lSamples, unsigned code[4]) {
  ReadFrame(lpBuffer, lStart);
}

////////////////////////////////////////////////////////////////////////
//////////// IAVIStream

STDMETHODIMP VapourSynthStream::Read(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples) {
  parent->Lock();
  HRESULT result = Read2(lStart, lSamples, lpBuffer, cbBuffer, plBytes, plSamples);
  parent->Unlock();
  return result;
}

HRESULT VapourSynthStream::Read2(LONG lStart, LONG lSamples, LPVOID lpBuffer, LONG cbBuffer, LONG *plBytes, LONG *plSamples) {
  int fp_state = _control87( 0, 0 );
  _control87( FP_STATE, 0xffffffff );
  unsigned code[4] = {0, 0, 0, 0};

	{
	if (lStart >= (parent->vi->numFrames ? parent->vi->numFrames : UNDEFINED_LENGTH)) {
      if (plSamples)
		  *plSamples = 0;
      if (plBytes)
		  *plBytes = 0;
      return S_OK;
    }

    int image_size = ImageSize(parent->vi);
    if (plSamples)
		*plSamples = 1;
    if (plBytes)
		*plBytes = image_size;

    if (!lpBuffer) {
      return S_OK;
    } else if (cbBuffer < image_size) {
      return AVIERR_BUFFERTOOSMALL;
    }
  }

  try {
      ReadHelper(lpBuffer, lStart, lSamples, code);
  }
  catch (...) {
    return E_FAIL;
  }
  return S_OK;
}

STDMETHODIMP VapourSynthStream::ReadFormat(LONG lPos, LPVOID lpFormat, LONG *lpcbFormat) {
  if (!lpcbFormat)
	  return E_POINTER;

  if (!lpFormat) {
    *lpcbFormat = sizeof(BITMAPINFOHEADER);
    return S_OK;
  }

  memset(lpFormat, 0, *lpcbFormat);

  const VSVideoInfo* const vi = parent->vi;

    BITMAPINFOHEADER bi;
    memset(&bi, 0, sizeof(bi));
    bi.biSize = sizeof(bi);
    bi.biWidth = vi->width;
    bi.biHeight = vi->height;
    bi.biPlanes = 1;
    bi.biBitCount = vi->format->bytesPerSample * 8;
    if (vi->format->numPlanes == 3)
        bi.biBitCount +=  (bi.biBitCount * 2) >> (vi->format->subSamplingH + vi->format->subSamplingW); //fixme, this is bullshit
    if (vi->format->id == pfCompatBGR32) 
	    bi.biCompression = BI_RGB;
    else if (vi->format->id == pfCompatYUY2) 
	    bi.biCompression = '2YUY';
    else if (vi->format->id == pfYUV420P8) 
	    bi.biCompression = '21VY';
    else if (vi->format->id == pfGray8) 
	    bi.biCompression = '008Y'; 
    else if (vi->format->id == pfYUV444P8) 
	    bi.biCompression = '42VY'; 
    else if (vi->format->id == pfYUV422P8) 
	    bi.biCompression = '61VY'; 
    else if (vi->format->id == pfYUV411P8) 
	    bi.biCompression = 'B14Y'; 
    else if (vi->format->id == pfYUV410P8) 
	    bi.biCompression = '9UVY'; 
	else {
	    _ASSERT(FALSE);
	}
	
	  bi.biSizeImage = ImageSize(vi);
    *lpcbFormat = min(*lpcbFormat, sizeof(bi));
    memcpy(lpFormat, &bi, size_t(*lpcbFormat));
  
  return S_OK;
}

STDMETHODIMP VapourSynthStream::Write(LONG lStart, LONG lSamples, LPVOID lpBuffer,
	LONG cbBuffer, DWORD dwFlags, LONG FAR *plSampWritten, 
    LONG FAR *plBytesWritten) {
	return AVIERR_READONLY;
}

