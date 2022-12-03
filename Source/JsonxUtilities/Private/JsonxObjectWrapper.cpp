// Copyright Epic Games, Inc. All Rights Reserved.

#include "JsonxObjectWrapper.h"
#include "Policies/CondensedJsonxPrintPolicy.h"
#include "Serialization/JsonxReader.h"
#include "Serialization/JsonxSerializer.h"

bool FJsonxObjectWrapper::ImportTextItem(const TCHAR*& Buffer, int32 PortFlags, UObject* Parent, FOutputDevice* ErrorText)
{
	// read JSONX string from Buffer
	FString Jsonx;
	if (*Buffer == TCHAR('"'))
	{
		int32 NumCharsRead = 0;
		if (!FParse::QuotedString(Buffer, Jsonx, &NumCharsRead))
		{
			ErrorText->Logf(ELogVerbosity::Warning, TEXT("FJsonxObjectWrapper::ImportTextItem: Bad quoted string: %s\n"), Buffer);
			return false;
		}
		Buffer += NumCharsRead;
	}
	else
	{
		// consume the rest of the string (this happens on Paste)
		Jsonx = Buffer;
		Buffer += Jsonx.Len();
	}

	// empty string yields empty shared pointer
	if (Jsonx.IsEmpty())
	{
		JsonxString.Empty();
		JsonxObject.Reset();
		return true;
	}

	// parse the json
	if (!JsonxObjectFromString(Jsonx))
	{
		if (ErrorText)
		{
			ErrorText->Logf(ELogVerbosity::Warning, TEXT("FJsonxObjectWrapper::ImportTextItem - Unable to parse json: %s\n"), *Jsonx);
		}
		return false;
	}
	JsonxString = Jsonx;
	return true;
}

bool FJsonxObjectWrapper::ExportTextItem(FString& ValueStr, FJsonxObjectWrapper const& DefaultValue, UObject* Parent, int32 PortFlags, UObject* ExportRootScope) const
{
	// empty pointer yields empty string
	if (!JsonxObject.IsValid())
	{
		ValueStr.Empty();
		return true;
	}

	// serialize the json
	return JsonxObjectToString(ValueStr);
}

void FJsonxObjectWrapper::PostSerialize(const FArchive& Ar)
{
	if (!JsonxString.IsEmpty())
	{
		// try to parse JsonxString
		if (!JsonxObjectFromString(JsonxString))
		{
			// do not abide a string that won't parse
			JsonxString.Empty();
		}
	}
}

bool FJsonxObjectWrapper::JsonxObjectToString(FString& Str) const
{
	TSharedRef<TJsonxWriter<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>> JsonxWriter = TJsonxWriterFactory<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>::Create(&Str, 0);
	return FJsonxSerializer::Serialize(JsonxObject.ToSharedRef(), JsonxWriter, true);
}

bool FJsonxObjectWrapper::JsonxObjectFromString(const FString& Str)
{
	TSharedRef<TJsonxReader<>> JsonxReader = TJsonxReaderFactory<>::Create(Str);
	return FJsonxSerializer::Deserialize(JsonxReader, JsonxObject);
}

