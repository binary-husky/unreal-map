// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GenericPlatform/GenericPlatformMath.h"
#include "Serialization/JsonxTypes.h"
#include "Policies/PrettyJsonxPrintPolicy.h"
#include "Serialization/MemoryWriter.h"

#define JSONX_LOW_PRECISION

/**
 * Takes an input string and escapes it so it can be written as a valid Jsonx string. Also adds the quotes.
 * Appends to a given string-like object to avoid reallocations.
 * String-like object must support operator+=(const TCHAR*) and operation+=(TCHAR)
 *
 * @param AppendTo the string to append to.
 * @param StringVal the string to escape
 * @return the AppendTo string for convenience.
 */
template<typename StringType>
inline StringType& AppendEscapeJsonxString(StringType& AppendTo, const FString& StringVal)
{
	AppendTo += TEXT("\"");
	for (const TCHAR* Char = *StringVal; *Char != TCHAR('\0'); ++Char)
	{
		switch (*Char)
		{
		case TCHAR('\\'): AppendTo += TEXT("\\\\"); break;
		case TCHAR('\n'): AppendTo += TEXT("\\n"); break;
		case TCHAR('\t'): AppendTo += TEXT("\\t"); break;
		case TCHAR('\b'): AppendTo += TEXT("\\b"); break;
		case TCHAR('\f'): AppendTo += TEXT("\\f"); break;
		case TCHAR('\r'): AppendTo += TEXT("\\r"); break;
		case TCHAR('\"'): AppendTo += TEXT("\\\""); break;
		default:
			// Must escape control characters
			if (*Char >= TCHAR(32))
			{
				AppendTo += *Char;
			}
			else
			{
				AppendTo.Appendf(TEXT("\\u%04x"), *Char);
			}
		}
	}
	AppendTo += TEXT("\"");

	return AppendTo;
}

/**
 * Takes an input string and escapes it so it can be written as a valid Jsonx string. Also adds the quotes.
 *
 * @param StringVal the string to escape
 * @return the given string, escaped to produce a valid Jsonx string.
 */
inline FString EscapeJsonxString(const FString& StringVal)
{
	FString Result;
	return AppendEscapeJsonxString(Result, StringVal);
}

/**
 * Template for Jsonx writers.
 *
 * @param CharType The type of characters to print, i.e. TCHAR or ANSICHAR.
 * @param PrintPolicy The print policy to use when writing the output string (default = TPrettyJsonxPrintPolicy).
 */
template <class CharType = TCHAR, class PrintPolicy = TPrettyJsonxPrintPolicy<CharType> >
class TJsonxWriter
{
public:

	static TSharedRef< TJsonxWriter > Create( FArchive* const Stream, int32 InitialIndentLevel = 0 )
	{
		return MakeShareable( new TJsonxWriter< CharType, PrintPolicy >( Stream, InitialIndentLevel ) );
	}

public:

	virtual ~TJsonxWriter() { }

	FORCEINLINE int32 GetIndentLevel() const { return IndentLevel; }

	bool CanWriteObjectStart() const
	{
		return CanWriteObjectWithoutIdentifier();
	}

	void WriteObjectStart()
	{
		check(CanWriteObjectWithoutIdentifier());
		if (PreviousTokenWritten != EJsonxToken::None )
		{
			WriteCommaIfNeeded();
		}

		if ( PreviousTokenWritten != EJsonxToken::None )
		{
			PrintPolicy::WriteLineTerminator(Stream);
			PrintPolicy::WriteTabs(Stream, IndentLevel);
		}

		PrintPolicy::WriteChar(Stream, CharType('{'));
		++IndentLevel;
		Stack.Push( EJsonx::Object );
		PreviousTokenWritten = EJsonxToken::CurlyOpen;
	}

	void WriteObjectStart( const FString& Identifier )
	{
		check( Stack.Top() == EJsonx::Object );
		WriteIdentifier( Identifier );

		PrintPolicy::WriteLineTerminator(Stream);
		PrintPolicy::WriteTabs(Stream, IndentLevel);
		PrintPolicy::WriteChar(Stream, CharType('{'));
		++IndentLevel;
		Stack.Push( EJsonx::Object );
		PreviousTokenWritten = EJsonxToken::CurlyOpen;
	}

	void WriteObjectEnd()
	{
		check( Stack.Top() == EJsonx::Object );

		PrintPolicy::WriteLineTerminator(Stream);

		--IndentLevel;
		PrintPolicy::WriteTabs(Stream, IndentLevel);
		PrintPolicy::WriteChar(Stream, CharType('}'));
		Stack.Pop();
		PreviousTokenWritten = EJsonxToken::CurlyClose;
	}

	void WriteArrayStart()
	{
		check(CanWriteValueWithoutIdentifier());
		if ( PreviousTokenWritten != EJsonxToken::None )
		{
			WriteCommaIfNeeded();
		}

		if ( PreviousTokenWritten != EJsonxToken::None )
		{
			PrintPolicy::WriteLineTerminator(Stream);
			PrintPolicy::WriteTabs(Stream, IndentLevel);
		}

		PrintPolicy::WriteChar(Stream, CharType('['));
		++IndentLevel;
		Stack.Push( EJsonx::Array );
		PreviousTokenWritten = EJsonxToken::SquareOpen;
	}

	void WriteArrayStart( const FString& Identifier )
	{
		check( Stack.Top() == EJsonx::Object );
		WriteIdentifier( Identifier );

		PrintPolicy::WriteSpace( Stream );
		PrintPolicy::WriteChar(Stream, CharType('['));
		++IndentLevel;
		Stack.Push( EJsonx::Array );
		PreviousTokenWritten = EJsonxToken::SquareOpen;
	}

	void WriteArrayEnd()
	{
		check( Stack.Top() == EJsonx::Array );

		--IndentLevel;
		if ( PreviousTokenWritten == EJsonxToken::SquareClose || PreviousTokenWritten == EJsonxToken::CurlyClose || PreviousTokenWritten == EJsonxToken::String )
		{
			PrintPolicy::WriteLineTerminator(Stream);
			PrintPolicy::WriteTabs(Stream, IndentLevel);
		}
		else if ( PreviousTokenWritten != EJsonxToken::SquareOpen )
		{
			PrintPolicy::WriteSpace( Stream );
		}

		PrintPolicy::WriteChar(Stream, CharType(']'));
		Stack.Pop();
		PreviousTokenWritten = EJsonxToken::SquareClose;
	}

	template <class FValue>
	void WriteValue(FValue Value)
	{
		check(CanWriteValueWithoutIdentifier());
		WriteCommaIfNeeded();

		if (PreviousTokenWritten == EJsonxToken::SquareOpen || EJsonxToken_IsShortValue(PreviousTokenWritten))
		{
			PrintPolicy::WriteSpace( Stream );
		}
		else
		{
			PrintPolicy::WriteLineTerminator(Stream);
			PrintPolicy::WriteTabs(Stream, IndentLevel);
		}

		PreviousTokenWritten = WriteValueOnly( Value );
	}

	void WriteValue(const FString& Value)
	{
		check(CanWriteValueWithoutIdentifier());
		WriteCommaIfNeeded();

		PrintPolicy::WriteLineTerminator(Stream);
		PrintPolicy::WriteTabs(Stream, IndentLevel);
		PreviousTokenWritten = WriteValueOnly(Value);
	}

	template <class FValue>
	void WriteValue(const FString& Identifier, FValue Value)
	{
		check( Stack.Top() == EJsonx::Object );
		WriteIdentifier( Identifier );

		PrintPolicy::WriteSpace(Stream);
		PreviousTokenWritten = WriteValueOnly(MoveTemp(Value));
	}

	template<class ElementType>
	void WriteValue(const FString& Identifier, const TArray<ElementType>& Array)
	{
		WriteArrayStart(Identifier);
		for (int Idx = 0; Idx < Array.Num(); Idx++)
		{
			WriteValue(Array[Idx]);
		}
		WriteArrayEnd();
	}

	void WriteValue(const FString& Identifier, const TCHAR* Value)
	{
		WriteValue(Identifier, FString(Value));
	}

	// WARNING: THIS IS DANGEROUS. Use this only if you know for a fact that the Value is valid JSONX!
	// Use this to insert the results of a different JSONX Writer in.
	void WriteRawJSONXValue( const FString& Identifier, const FString& Value )
	{
		check( Stack.Top() == EJsonx::Object );
		WriteIdentifier( Identifier );

		PrintPolicy::WriteSpace(Stream);
		PrintPolicy::WriteString(Stream, Value);
		PreviousTokenWritten = EJsonxToken::String;
	}

	void WriteNull( const FString& Identifier )
	{
		WriteValue(Identifier, nullptr);
	}

	void WriteValue( const TCHAR* Value )
	{
		WriteValue(FString(Value));
	}

	// WARNING: THIS IS DANGEROUS. Use this only if you know for a fact that the Value is valid JSONX!
	// Use this to insert the results of a different JSONX Writer in.
	void WriteRawJSONXValue( const FString& Value )
	{
		check(CanWriteValueWithoutIdentifier());
		WriteCommaIfNeeded();

		if ( PreviousTokenWritten != EJsonxToken::True && PreviousTokenWritten != EJsonxToken::False && PreviousTokenWritten != EJsonxToken::SquareOpen )
		{
			PrintPolicy::WriteLineTerminator(Stream);
			PrintPolicy::WriteTabs(Stream, IndentLevel);
		}
		else
		{
			PrintPolicy::WriteSpace( Stream );
		}

		PrintPolicy::WriteString(Stream, Value);
		PreviousTokenWritten = EJsonxToken::String;
	}

	void WriteNull()
	{
		WriteValue(nullptr);
	}

	virtual bool Close()
	{
		return ( PreviousTokenWritten == EJsonxToken::None ||
				 PreviousTokenWritten == EJsonxToken::CurlyClose  ||
				 PreviousTokenWritten == EJsonxToken::SquareClose )
				&& Stack.Num() == 0;
	}

	/**
	 * WriteValue("Foo", Bar) should be equivalent to WriteIdentifierPrefix("Foo"), WriteValue(Bar)
	 */
	void WriteIdentifierPrefix(const FString& Identifier)
	{
		check(Stack.Top() == EJsonx::Object);
		WriteIdentifier(Identifier);
		PrintPolicy::WriteSpace(Stream);
		PreviousTokenWritten = EJsonxToken::Identifier;
	}

protected:

	/**
	 * Creates and initializes a new instance.
	 *
	 * @param InStream An archive containing the input.
	 * @param InitialIndentLevel The initial indentation level.
	 */
	TJsonxWriter( FArchive* const InStream, int32 InitialIndentLevel )
		: Stream( InStream )
		, Stack()
		, PreviousTokenWritten(EJsonxToken::None)
		, IndentLevel(InitialIndentLevel)
	{ }

protected:

	FORCEINLINE bool CanWriteValueWithoutIdentifier() const
	{
		return Stack.Num() <= 0 || Stack.Top() == EJsonx::Array || PreviousTokenWritten == EJsonxToken::Identifier;
	}

	FORCEINLINE bool CanWriteObjectWithoutIdentifier() const
	{
		return Stack.Num() <= 0 || Stack.Top() == EJsonx::Array || PreviousTokenWritten == EJsonxToken::Identifier || PreviousTokenWritten == EJsonxToken::Colon;
	}

	FORCEINLINE void WriteCommaIfNeeded()
	{
		if ( PreviousTokenWritten != EJsonxToken::CurlyOpen && PreviousTokenWritten != EJsonxToken::SquareOpen && PreviousTokenWritten != EJsonxToken::Identifier)
		{
			PrintPolicy::WriteChar(Stream, CharType(','));
		}
	}

	FORCEINLINE void WriteIdentifier( const FString& Identifier )
	{
		WriteCommaIfNeeded();
		PrintPolicy::WriteLineTerminator(Stream);

		PrintPolicy::WriteTabs(Stream, IndentLevel);
		WriteStringValue( Identifier );
		PrintPolicy::WriteChar(Stream, CharType(':'));
	}

	FORCEINLINE EJsonxToken WriteValueOnly(bool Value)
	{
		PrintPolicy::WriteString(Stream, Value ? TEXT("true") : TEXT("false"));
		return Value ? EJsonxToken::True : EJsonxToken::False;
	}

#ifdef JSONX_LOW_PRECISION

	FORCEINLINE EJsonxToken WriteValueOnly(float Value)
	{
		if (FGenericPlatformMath::IsFinite(Value))
		{
			PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%.7g"), Value));
		}
		else
		{
			PrintPolicy::WriteString(Stream, FString::Printf(TEXT("Infinity")));
		}
		//PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%g"), Value));
		return EJsonxToken::Number;
	}

	FORCEINLINE EJsonxToken WriteValueOnly(double Value)
	{
		// Specify 17 significant digits, the most that can ever be useful from a double
		// In particular, this ensures large integers are written correctly
		if (FGenericPlatformMath::IsFinite(Value))
		{
			PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%.7g"), Value));
		}
		else
		{
			PrintPolicy::WriteString(Stream, FString::Printf(TEXT("Infinity")));
		}

		return EJsonxToken::Number;
	}

#else
	FORCEINLINE EJsonxToken WriteValueOnly(float Value)
	{

		PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%g"), Value));
		return EJsonxToken::Number;
	}

	FORCEINLINE EJsonxToken WriteValueOnly(double Value)
	{
		// Specify 17 significant digits, the most that can ever be useful from a double
		// In particular, this ensures large integers are written correctly

		PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%.17g"), Value));
		return EJsonxToken::Number;
	}
#endif




	FORCEINLINE EJsonxToken WriteValueOnly(int32 Value)
	{
		return WriteValueOnly((int64)Value);
	}

	FORCEINLINE EJsonxToken WriteValueOnly(int64 Value)
	{
		PrintPolicy::WriteString(Stream, FString::Printf(TEXT("%lld"), Value));
		return EJsonxToken::Number;
	}

	FORCEINLINE EJsonxToken WriteValueOnly(TYPE_OF_NULLPTR)
	{
		PrintPolicy::WriteString(Stream, TEXT("null"));
		return EJsonxToken::Null;
	}

	FORCEINLINE EJsonxToken WriteValueOnly(const FString& Value)
	{
		WriteStringValue(Value);
		return EJsonxToken::String;
	}

	virtual void WriteStringValue( const FString& String )
	{
		FString OutString = EscapeJsonxString(String);
		PrintPolicy::WriteString(Stream, OutString);
	}

	FArchive* const Stream;
	TArray<EJsonx> Stack;
	EJsonxToken PreviousTokenWritten;
	int32 IndentLevel;
};


template <class PrintPolicy = TPrettyJsonxPrintPolicy<TCHAR>>
class TJsonxStringWriter
	: public TJsonxWriter<TCHAR, PrintPolicy>
{
public:

	static TSharedRef<TJsonxStringWriter> Create( FString* const InStream, int32 InitialIndent = 0 )
	{
		return MakeShareable(new TJsonxStringWriter(InStream, InitialIndent));
	}

public:

	virtual ~TJsonxStringWriter()
	{
		check(this->Stream->Close());
		delete this->Stream;
	}

	virtual bool Close() override
	{
		FString Out;

		for (int32 i = 0; i < Bytes.Num(); i+=sizeof(TCHAR))
		{
			TCHAR* Char = static_cast<TCHAR*>(static_cast<void*>(&Bytes[i]));
			Out += *Char;
		}

		*OutString = Out;

		return TJsonxWriter<TCHAR, PrintPolicy>::Close();
	}

protected:

	TJsonxStringWriter( FString* const InOutString, int32 InitialIndent )
		: TJsonxWriter<TCHAR, PrintPolicy>(new FMemoryWriter(Bytes), InitialIndent)
		, Bytes()
		, OutString(InOutString)
	{ }

private:

	TArray<uint8> Bytes;
	FString* OutString;
};


template <class CharType = TCHAR, class PrintPolicy = TPrettyJsonxPrintPolicy<CharType>>
class TJsonxWriterFactory
{
public:

	static TSharedRef<TJsonxWriter<CharType, PrintPolicy>> Create( FArchive* const Stream, int32 InitialIndent = 0 )
	{
		return TJsonxWriter< CharType, PrintPolicy >::Create(Stream, InitialIndent);
	}

	static TSharedRef<TJsonxWriter<TCHAR, PrintPolicy>> Create( FString* const Stream, int32 InitialIndent = 0 )
	{
		return StaticCastSharedRef<TJsonxWriter<TCHAR, PrintPolicy>>(TJsonxStringWriter<PrintPolicy>::Create(Stream, InitialIndent));
	}
};
