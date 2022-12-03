// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/JsonxPrintPolicy.h"

/**
 * Template for print policies that generate human readable output.
 *
 * @param CharType The type of characters to print, i.e. TCHAR or ANSICHAR.
 */
template <class CharType>
struct TPrettyJsonxPrintPolicy
	: public TJsonxPrintPolicy<CharType>
{
	static inline void WriteLineTerminator( FArchive* Stream )
	{
		TJsonxPrintPolicy<CharType>::WriteString(Stream, LINE_TERMINATOR);
	}

	static inline void WriteTabs( FArchive* Stream, int32 Count )
	{
		CharType Tab = CharType('\t');

		for (int32 i = 0; i < Count; ++i)
		{
			TJsonxPrintPolicy<CharType>::WriteChar(Stream, Tab);
		}
	}

	static inline void WriteSpace( FArchive* Stream )
	{
		TJsonxPrintPolicy<CharType>::WriteChar(Stream, CharType(' '));
	}
};
