// Fill out your copyright notice in the Description page of Project Settings.


#include "HmpPythonIO.h"

// Sets default values
AHmpPythonIO::AHmpPythonIO()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

void AHmpPythonIO::BeginDestroy()
{
	Super::BeginDestroy();
	if (TcpAccpetedSocket)
	{
		TcpAccpetedSocket->Close();
	}
	if (ListenSocket)
	{
		ListenSocket->Close();
	}
}


void AHmpPythonIO::StartTcpServer(int32 InListenPort)
{
	FIPv4Address Address;
	FIPv4Address::Parse(TEXT("0.0.0.0"), Address);

	//Create Socket
	FIPv4Endpoint Endpoint(Address, InListenPort);
	FString ListenSocketName = TEXT("ue4-tcp-server");
	ListenSocket = FTcpSocketBuilder(*ListenSocketName)
		.AsBlocking()
		.AsReusable()
		.BoundToEndpoint(Endpoint);
	int32 setBufferSizeRes = 0;
	ListenSocket->SetReceiveBufferSize(ReceiveBufferSize, setBufferSizeRes);
	ListenSocket->SetSendBufferSize(SendBufferSize, setBufferSizeRes);
	ListenSocket->Listen(8);

}


void AHmpPythonIO::TcpServerSendJson(TSharedPtr<FJsonxObject> ReplyJson, float& encTime, float& sendTime)
{
	double t1 = FPlatformTime::Seconds();

	FString ReplyString;
	auto Writer = TJsonxWriterFactory<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>::Create(&ReplyString);
	FJsonxSerializer::Serialize(ReplyJson.ToSharedRef(), Writer);
	TCHAR* serializedChar = ReplyString.GetCharArray().GetData();
	FTCHARToUTF8 Converted(serializedChar);
	uint8* sendbuf = (uint8*)Converted.Get();
	int32 buffsize_raw = Converted.Length();

	// use XLZ4 built in unreal engine core to compress data before send, reducing IPC cost
	int dstSize = XLZ4_compress_fast((const char*)sendbuf, (char*)SendBuffer, buffsize_raw, SendBufferSize, 1);
	buffsize_raw = dstSize;

	double t2 = FPlatformTime::Seconds();


	// add checkEOF
	SendBuffer[buffsize_raw]     = 0xaa;
	SendBuffer[buffsize_raw + 1] = 0x55;
	SendBuffer[buffsize_raw + 2] = 0xaa;
	SendBuffer[buffsize_raw + 3] = 'H';
	SendBuffer[buffsize_raw + 4] = 'M';
	SendBuffer[buffsize_raw + 5] = 'P';
	SendBuffer[buffsize_raw + 6] = 0xaa;
	SendBuffer[buffsize_raw + 7] = 0x55;
	SendBuffer[buffsize_raw + 8] = '\0';

	ensureMsgf((buffsize_raw + 8 < SendBufferSize), TEXT("send buffer overflow!"));	// prevent buffer overflow!
	SendBufferUsage = (float)(buffsize_raw + 8) / SendBufferSize;
	int32 BytesSent;



	TcpAccpetedSocket->Send(SendBuffer, buffsize_raw + 8, BytesSent);	// tcp send
	ensureMsgf((BytesSent == buffsize_raw + 8), TEXT("send buffer incomplete!"));

	double t3 = FPlatformTime::Seconds();
	encTime = t2 - t1;
	sendTime = t3 - t2;
}

void AHmpPythonIO::TcpServerWaitClient()
{
	bool bHasPendingConnection = false;
	FTimespan timespan = FTimespan(0, 10, 0);
	ListenSocket->WaitForPendingConnection(bHasPendingConnection, timespan);
	TSharedPtr<FInternetAddr> Addr = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
	TcpAccpetedSocket = ListenSocket->Accept(*Addr, TEXT("tcp-client"));

}


bool AHmpPythonIO::TcpServerHasWaitingClient()
{
	bool bHasPendingConnection = false;
	ListenSocket->HasPendingConnection(bHasPendingConnection);
	return bHasPendingConnection;

}

bool AHmpPythonIO::TcpServerHasRecvData()
{
	uint32 BytesPendingdata = 0;
	return TcpAccpetedSocket->HasPendingData(BytesPendingdata);
}

FString AHmpPythonIO::TcpServerRecvBlocked(bool checkEOF, float& tcpWaitTime, float &decodeTime)
{
	ESocketConnectionState ConnectionState = TcpAccpetedSocket->GetConnectionState();
	if (ConnectionState != ESocketConnectionState::SCS_Connected)
	{
		return "";
	}
	int32 currenthead = 0;
	while (1)
	{
		int32 BytesRead = 0;
		uint32 BytesPendingdata = 0;
		double t1 = FPlatformTime::Seconds();
		// 24小时，有时候测试之间的间隔很长，长达1小时也是有可能的
		TcpAccpetedSocket->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromHours(24));
		double t2 = FPlatformTime::Seconds();
		tcpWaitTime = t2 - t1;
		if (!TcpAccpetedSocket->HasPendingData(BytesPendingdata)) {
			ensureMsgf(false, TEXT("Tcp wait timeout! The uhmap will exit now because nothing plays with it in 24 hours!"));
			return "";
		}
		TcpAccpetedSocket->Recv(&RecvBuffer[currenthead], ReceiveBufferSize, BytesRead);
		currenthead = currenthead + BytesRead;
		ensureMsgf((currenthead + 1 < ReceiveBufferSize), TEXT("recv buffer overflow"));	// Buffer overflow!
		RecvBufferUsage = (float)(currenthead + 1) / ReceiveBufferSize;
		RecvBuffer[currenthead] = '\0'; // seal the tail of buffer with '\0'
		if (checkEOF) 
		{
			if (currenthead >= 8 &&
				RecvBuffer[currenthead - 1] == 0x55 &&
				RecvBuffer[currenthead - 2] == 0xaa &&
				RecvBuffer[currenthead - 3] == 'P' &&
				RecvBuffer[currenthead - 4] == 'M' &&
				RecvBuffer[currenthead - 5] == 'H' &&
				RecvBuffer[currenthead - 6] == 0xaa &&
				RecvBuffer[currenthead - 7] == 0x55 &&
				RecvBuffer[currenthead - 8] == 0xaa)
			{
				// data ends with pre-defined marker, clear the marker, convert to String
				RecvBuffer[currenthead - 8] = '\0';
				RecvBuffer[currenthead - 7] = '\0';
				RecvBuffer[currenthead - 6] = '\0';
				RecvBuffer[currenthead - 5] = '\0';
				RecvBuffer[currenthead - 4] = '\0';
				RecvBuffer[currenthead - 3] = '\0';
				RecvBuffer[currenthead - 2] = '\0';
				RecvBuffer[currenthead - 1] = '\0';

				// decompress
				int dst_read = XLZ4_decompress_safe((const char*)RecvBuffer, (char*)RecvDecompressBuffer, currenthead - 8, ReceiveBufferSize);
				RecvDecompressBuffer[dst_read] = '\0'; // seal the top of buffer

				FString RecvString = FString(UTF8_TO_TCHAR(RecvDecompressBuffer));
				return RecvString;
			}
		}
		else {
			ensureMsgf(false, TEXT("must use checkEOF!"));	// prevent buffer overflow!
			FString RecvString = FString(UTF8_TO_TCHAR(RecvBuffer));
			return RecvString;
		}
		double t3 = FPlatformTime::Seconds();
		decodeTime = t3 - t2;
	}
}

FParsedDataInput AHmpPythonIO::ParsedTcpInData(FString TcpLatestRecvString)
{
	static const struct FParsedDataInput EmptyStruct;
	FParsedDataInput TcpParsedTcpInData = EmptyStruct;
	try 
	{

		TSharedRef<TJsonxReader<TCHAR>> JsonReader = TJsonxReaderFactory<TCHAR>::Create(TcpLatestRecvString);
		TSharedPtr<FJsonxObject> JsonObject = MakeShareable(new FJsonxObject);
		FJsonxSerializer::Deserialize(JsonReader, JsonObject);
		FJsonxObjectConverter::JsonxObjectToUStruct<FParsedDataInput>(JsonObject.ToSharedRef(), &TcpParsedTcpInData, 0, 0);
		TcpParsedTcpInData.valid = true;
		return TcpParsedTcpInData;

	}
	catch (...) 
	{

		TcpParsedTcpInData.valid = false;
		return TcpParsedTcpInData;

	}


}


void AHmpPythonIO::ConvertOutDataToJsonAndSendTcp(TArray<FAgentDataOutput> TcpOutDataArr, 
	FGlobalDataOutput GlobalData, float &toJsonTime, float &encTime, float &sendTime)
{
	double t1 = FPlatformTime::Seconds();

	TSharedRef<FJsonxObject> JsonReply = MakeShareable(new FJsonxObject);
	FAgentDataOutputArr Reply;
	Reply.Valid = true;
	Reply.DataArr = TcpOutDataArr;
	Reply.DataGlobal = GlobalData;
	FJsonxObjectConverter::UStructToJsonxObject(FAgentDataOutputArr::StaticStruct(), &Reply, JsonReply, 0, 0);
	toJsonTime = FPlatformTime::Seconds() - t1;
	//const FString t = FString("DataArr");
	auto J_DataArr = JsonReply->TryGetField("dataArr")->AsArray();

	TArray<TSharedPtr<FJsonxValue>> J_DataArrNew;

	for (int i = 0; i < J_DataArr.Num(); i++) 
	{
		auto agentJsonDataObj = J_DataArr[i]->AsObject();
		bool alive = true;
		agentJsonDataObj->TryGetBoolField("agentAlive", alive);
		if (!alive) 
		{
			TSharedPtr<FJsonxObject> replacement = MakeShareable(new FJsonxObject);
			// five core property
			replacement->SetBoolField("valid", true);
			replacement->SetBoolField("agentAlive", false);
			replacement->SetNumberField("agentTeam", (int)(agentJsonDataObj->GetNumberField("agentTeam")));
			replacement->SetNumberField("indexInTeam", (int)(agentJsonDataObj->GetNumberField("indexInTeam")));
			replacement->SetNumberField("uID", (int)(agentJsonDataObj->GetNumberField("uID") ));
			J_DataArrNew.Add(MakeShareable(new FJsonxValueObject(replacement)));
		}
		else 
		{
			J_DataArrNew.Add(J_DataArr[i]);
		}
	}
	JsonReply->SetArrayField("dataArr", J_DataArrNew);


	//for (auto t : J_DataArr->AsArray()) 
	//{
	//	auto tt = t->AsObject();
	//	if (tt) {
	//		bool alive = true;
	//		if (tt->TryGetBoolField("agentAlive", alive)) {
	//			if (!alive) {
	//				TSharedPtr<FJsonxObject> replacement = MakeShareable(new FJsonxObject);
	//				// five core property
	//				replacement->SetBoolField("valid", true);
	//				replacement->SetBoolField("agentAlive", false);
	//				replacement->SetNumberField("agentTeam", (int)(tt->GetNumberField("agentTeam")));
	//				replacement->SetNumberField("indexInTeam", (int)(tt->GetNumberField("indexInTeam")));
	//				replacement->SetNumberField("uID", (int)(tt->GetNumberField("uID") ));
	//				*tt = *replacement;
	//			}
	//		}
	//	}
	//}



	TcpServerSendJson(JsonReply, encTime, sendTime);
}

void AHmpPythonIO::sleep_thread(float second)
{
	FPlatformProcess::Sleep(second);
}

void AHmpPythonIO::ChangeEngineFixedFrameRate(float fps)
{
	if (fps >= 2 && fps <= 5000)
	{
		GEngine->FixedFrameRate = fps;
	}
}

void AHmpPythonIO::DisableRendering()
{
	ensureMsgf(false, TEXT("do not use this func!"));
	GEngine->GameViewport->bDisableWorldRendering = false;
}

void AHmpPythonIO::EnableRendering()
{
	ensureMsgf(false, TEXT("do not use this func!"));
	GEngine->GameViewport->bDisableWorldRendering = true;
}


void AHmpPythonIO::tic()
{
	tic_second = FPlatformTime::Seconds();
}

float AHmpPythonIO::toc()
{
	//FPlatformProcess::Sleep(0.05);
	toc_second = FPlatformTime::Seconds();
	return toc_second - tic_second;
}

void AHmpPythonIO::dur_tic()
{
	dur_tic_second = FPlatformTime::Seconds();
}
void AHmpPythonIO::dur_toc()
{
	dur_toc_second = FPlatformTime::Seconds();
	dur_sum_second += dur_toc_second - dur_tic_second;
}
float AHmpPythonIO::dur_reset()
{
	float tmp = dur_sum_second;
	dur_sum_second = 0;
	return tmp;
}



void AHmpPythonIO::RaiseErrorNative(UObject* WorldContextObject, const FString& ErrorMessage, bool bPrintToOutputLog)
{
	FString MessageToLog = FString::Printf(TEXT("\"%s\""), *ErrorMessage);


#if WITH_EDITOR
	/*FKismetDebugUtilities::OnScriptException(WorldContextObject,);*/
	TSharedRef<FTokenizedMessage> TokenizedMessage = FTokenizedMessage::Create(EMessageSeverity::Error, FText::FromString(MessageToLog));
	TokenizedMessage.Get().AddToken(FUObjectToken::Create(WorldContextObject));
	FMessageLog BlueprintLog = FMessageLog("BlueprintLog").SuppressLoggingToOutputLog(!bPrintToOutputLog);
	BlueprintLog.AddMessage(TokenizedMessage);
	BlueprintLog.Notify();

#else
	if (bPrintToOutputLog)
	{
		UE_LOG(LogTemp, Error, TEXT("%s"), *MessageToLog);
	}
#endif
}

void AHmpPythonIO::exit_hmp(bool force)
{
	FGenericPlatformMisc::RequestExit(force);
}