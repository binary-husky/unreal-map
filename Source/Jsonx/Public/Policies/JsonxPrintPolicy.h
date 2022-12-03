// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

/**
 * Base template for Jsonx print policies.
 *
 * @param CharType The type of characters to print, i.e. TCHAR or ANSICHAR.
 */
template <class CharType>
struct TJsonxPrintPolicy
{
	/**
	 * Writes a single character to the output stream.
	 *
	 * @param Stream The stream to write to.
	 * @param Char The character to write.
	 */
	static inline void WriteChar( FArchive* Stream, CharType Char )
	{
		Stream->Serialize(&Char, sizeof(CharType));
	}

	/**
	 * Writes a string to the output stream.
	 *
	 * @param Stream The stream to write to.
	 * @param String The string to write.
	 */
	static inline void WriteString( FArchive* Stream, const FString& String )
	{
		const TCHAR* CharPtr = *String;

		for (int32 CharIndex = 0; CharIndex < String.Len(); ++CharIndex, ++CharPtr)
		{
			WriteChar(Stream, *CharPtr);
		}
	}
};


/**
 * Specialization for TCHAR that allows direct copying from FString data.
 */
template <>
inline void TJsonxPrintPolicy<TCHAR>::WriteString( FArchive* Stream, const FString& String )
{
	Stream->Serialize((void*)*String, String.Len() * sizeof(TCHAR));
}


#if !PLATFORM_TCHAR_IS_CHAR16

/**
 * Specialization for UTF16CHAR that writes FString data UTF-16.
 */
template <>
inline void TJsonxPrintPolicy<UTF16CHAR>::WriteString(FArchive* Stream, const FString& String)
{
	// Note: This is a no-op on platforms that are using a 16-bit TCHAR
	FTCHARToUTF16 UTF16String(*String, String.Len());

	Stream->Serialize((void*)UTF16String.Get(), UTF16String.Length() * sizeof(UTF16CHAR));
}

#endif

