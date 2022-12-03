// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Serialization/JsonxTypes.h"
#include "Serialization/BufferReader.h"

class Error;

#define JSONX_NOTATIONMAP_DEF \
static EJsonxNotation TokenToNotationTablex[] =  \
{ \
	EJsonxNotation::Error,			/*EJsonxToken::None*/ \
	EJsonxNotation::Error,			/*EJsonxToken::Comma*/ \
	EJsonxNotation::ObjectStart,		/*EJsonxToken::CurlyOpen*/ \
	EJsonxNotation::ObjectEnd,		/*EJsonxToken::CurlyClose*/ \
	EJsonxNotation::ArrayStart,		/*EJsonxToken::SquareOpen*/ \
	EJsonxNotation::ArrayEnd,		/*EJsonxToken::SquareClose*/ \
	EJsonxNotation::Error,			/*EJsonxToken::Colon*/ \
	EJsonxNotation::String,			/*EJsonxToken::String*/ \
	EJsonxNotation::Number,			/*EJsonxToken::Number*/ \
	EJsonxNotation::Boolean,			/*EJsonxToken::True*/ \
	EJsonxNotation::Boolean,			/*EJsonxToken::False*/ \
	EJsonxNotation::Null,			/*EJsonxToken::Null*/ \
};

#ifndef WITH_JSONX_INLINED_NOTATIONMAP
#define WITH_JSONX_INLINED_NOTATIONMAP 0
#endif // WITH_JSONX_INLINED_NOTATIONMAP

#if !WITH_JSONX_INLINED_NOTATIONMAP
JSONX_NOTATIONMAP_DEF;
#endif // WITH_JSONX_INLINED_NOTATIONMAP

template <class CharType = TCHAR>
class TJsonxReader
{
public:

	static TSharedRef< TJsonxReader<CharType> > Create( FArchive* const Stream )
	{
		return MakeShareable( new TJsonxReader<CharType>( Stream ) );
	}

public:

	virtual ~TJsonxReader() {}

	bool ReadNext( EJsonxNotation& Notation )
	{
		if (!ErrorMessage.IsEmpty())
		{
			Notation = EJsonxNotation::Error;
			return false;
		}

		if (Stream == nullptr)
		{
			Notation = EJsonxNotation::Error;
			SetErrorMessage(TEXT("Null Stream"));
			return true;
		}

		const bool AtEndOfStream = Stream->AtEnd();

		if (AtEndOfStream && !FinishedReadingRootObject)
		{
			Notation = EJsonxNotation::Error;
			SetErrorMessage(TEXT("Improperly formatted."));
			return true;
		}

		if (FinishedReadingRootObject && !AtEndOfStream)
		{
			Notation = EJsonxNotation::Error;
			SetErrorMessage(TEXT("Unexpected additional input found."));
			return true;
		}

		if (AtEndOfStream)
		{
			return false;
		}

		bool ReadWasSuccess = false;
		Identifier.Empty();

		do
		{
			EJsonx CurrentState = EJsonx::None;

			if (ParseState.Num() > 0)
			{
				CurrentState = ParseState.Top();
			}

			switch (CurrentState)
			{
				case EJsonx::Array:
					ReadWasSuccess = ReadNextArrayValue( /*OUT*/ CurrentToken );
					break;

				case EJsonx::Object:
					ReadWasSuccess = ReadNextObjectValue( /*OUT*/ CurrentToken );
					break;

				default:
					ReadWasSuccess = ReadStart( /*OUT*/ CurrentToken );
					break;
			}
		}
		while (ReadWasSuccess && (CurrentToken == EJsonxToken::None));

#if WITH_JSONX_INLINED_NOTATIONMAP
		JSONX_NOTATIONMAP_DEF;
#endif // WITH_JSONX_INLINED_NOTATIONMAP

		Notation = TokenToNotationTablex[(int32)CurrentToken];
		FinishedReadingRootObject = ParseState.Num() == 0;

		if (!ReadWasSuccess || (Notation == EJsonxNotation::Error))
		{
			Notation = EJsonxNotation::Error;

			if (ErrorMessage.IsEmpty())
			{
				SetErrorMessage(TEXT("Unknown Error Occurred"));
			}

			return true;
		}

		if (FinishedReadingRootObject && !Stream->AtEnd())
		{
			ReadWasSuccess = ParseWhiteSpace();
		}

		return ReadWasSuccess;
	}

	bool SkipObject()
	{
		return ReadUntilMatching(EJsonxNotation::ObjectEnd);
	}

	bool SkipArray()
	{
		return ReadUntilMatching(EJsonxNotation::ArrayEnd);
	}

	FORCEINLINE virtual const FString& GetIdentifier() const { return Identifier; }

	FORCEINLINE virtual  const FString& GetValueAsString() const 
	{ 
		check(CurrentToken == EJsonxToken::String);
		return StringValue;
	}
	
	FORCEINLINE double GetValueAsNumber() const 
	{ 
		check(CurrentToken == EJsonxToken::Number);
		return NumberValue;
	}

	FORCEINLINE const FString& GetValueAsNumberString() const
	{
		check(CurrentToken == EJsonxToken::Number);
		return StringValue;
	}
	
	FORCEINLINE bool GetValueAsBoolean() const 
	{ 
		check((CurrentToken == EJsonxToken::True) || (CurrentToken == EJsonxToken::False));
		return BoolValue; 
	}

	FORCEINLINE const FString& GetErrorMessage() const
	{
		return ErrorMessage;
	}

	FORCEINLINE const uint32 GetLineNumber() const
	{
		return LineNumber;
	}

	FORCEINLINE const uint32 GetCharacterNumber() const
	{
		return CharacterNumber;
	}

protected:

	/** Hidden default constructor. */
	TJsonxReader()
		: ParseState()
		, CurrentToken( EJsonxToken::None )
		, Stream( nullptr )
		, Identifier()
		, ErrorMessage()
		, StringValue()
		, NumberValue( 0.0f )
		, LineNumber( 1 )
		, CharacterNumber( 0 )
		, BoolValue( false )
		, FinishedReadingRootObject( false )
	{ }

	/**
	 * Creates and initializes a new instance with the given input.
	 *
	 * @param InStream An archive containing the input.
	 */
	TJsonxReader(FArchive* InStream)
		: ParseState()
		, CurrentToken(EJsonxToken::None)
		, Stream(InStream)
		, Identifier()
		, ErrorMessage()
		, StringValue()
		, NumberValue(0.0f)
		, LineNumber(1)
		, CharacterNumber(0)
		, BoolValue(false)
		, FinishedReadingRootObject(false)
	{ }

private:

	void SetErrorMessage( const FString& Message )
	{
		ErrorMessage = Message + FString::Printf(TEXT(" Line: %u Ch: %u"), LineNumber, CharacterNumber);
	}

	bool ReadUntilMatching( const EJsonxNotation ExpectedNotation )
	{
		uint32 ScopeCount = 0;
		EJsonxNotation Notation;

		while (ReadNext(Notation))
		{
			if ((ScopeCount == 0) && (Notation == ExpectedNotation))
			{
				return true;
			}

			switch (Notation)
			{
			case EJsonxNotation::ObjectStart:
			case EJsonxNotation::ArrayStart:
				++ScopeCount;
				break;

			case EJsonxNotation::ObjectEnd:
			case EJsonxNotation::ArrayEnd:
				--ScopeCount;
				break;

			case EJsonxNotation::Boolean:
			case EJsonxNotation::Null:
			case EJsonxNotation::Number:
			case EJsonxNotation::String:
				break;

			case EJsonxNotation::Error:
				return false;
				break;
			}
		}

		return !Stream->IsError();
	}

	bool ReadStart( EJsonxToken& Token )
	{
		if (!ParseWhiteSpace())
		{
			return false;
		}

		Token = EJsonxToken::None;

		if (NextToken(Token) == false)
		{
			return false;
		}

		if ((Token != EJsonxToken::CurlyOpen) && (Token != EJsonxToken::SquareOpen))
		{
			SetErrorMessage(TEXT("Open Curly or Square Brace token expected, but not found."));
			return false;
		}

		return true;
	}

	bool ReadNextObjectValue( EJsonxToken& Token )
	{
		const bool bCommaPrepend = Token != EJsonxToken::CurlyOpen;
		Token = EJsonxToken::None;

		if (NextToken(Token) == false)
		{
			return false;
		}

		if (Token == EJsonxToken::CurlyClose)
		{
			return true;
		}
		else
		{
			if (bCommaPrepend)
			{
				if (Token != EJsonxToken::Comma)
				{
					SetErrorMessage( TEXT("Comma token expected, but not found.") );
					return false;
				}

				Token = EJsonxToken::None;

				if (!NextToken(Token))
				{
					return false;
				}
			}

			if (Token != EJsonxToken::String)
			{
				SetErrorMessage( TEXT("String token expected, but not found.") );
				return false;
			}

			Identifier = StringValue;
			Token = EJsonxToken::None;

			if (!NextToken(Token))
			{
				return false;
			}

			if (Token != EJsonxToken::Colon)
			{
				SetErrorMessage( TEXT("Colon token expected, but not found.") );
				return false;
			}

			Token = EJsonxToken::None;

			if (!NextToken(Token))
			{
				return false;
			}
		}

		return true;
	}

	bool ReadNextArrayValue( EJsonxToken& Token )
	{
		const bool bCommaPrepend = Token != EJsonxToken::SquareOpen;

		Token = EJsonxToken::None;

		if (!NextToken(Token))
		{
			return false;
		}

		if (Token == EJsonxToken::SquareClose)
		{
			return true;
		}
		else
		{
			if (bCommaPrepend)
			{
				if (Token != EJsonxToken::Comma)
				{
					SetErrorMessage( TEXT("Comma token expected, but not found.") );
					return false;
				}

				Token = EJsonxToken::None;

				if (!NextToken(Token))
				{
					return false;
				}
			}
		}

		return true;
	}

	bool NextToken( EJsonxToken& OutToken )
	{
		while (!Stream->AtEnd())
		{
			CharType Char;
			if (!Serialize(&Char, sizeof(CharType)))
			{
				return false;
			}
			++CharacterNumber;

			if (Char == CharType('\0'))
			{
				break;
			}

			if (IsLineBreak(Char))
			{
				++LineNumber;
				CharacterNumber = 0;
			}

			if (!IsWhitespace(Char))
			{
				if (IsJsonxNumber(Char))
				{
					if (!ParseNumberToken(Char))
					{
						return false;
					}

					OutToken = EJsonxToken::Number;
					return true;
				}

				switch (Char)
				{
				case CharType('{'):
					OutToken = EJsonxToken::CurlyOpen; ParseState.Push( EJsonx::Object );
					return true;

				case CharType('}'):
					{
						OutToken = EJsonxToken::CurlyClose;
						if (ParseState.Num())
						{
							ParseState.Pop();
							return true;
						}
						else
						{
							SetErrorMessage(TEXT("Unknown state reached while parsing Jsonx token."));
							return false;
						}
					}

				case CharType('['):
					OutToken = EJsonxToken::SquareOpen; ParseState.Push( EJsonx::Array );
					return true;

				case CharType(']'):
					{
						OutToken = EJsonxToken::SquareClose;
						if (ParseState.Num())
						{
							ParseState.Pop();
							return true;
						}
						else
						{
							SetErrorMessage(TEXT("Unknown state reached while parsing Jsonx token."));
							return false;
						}
					}

				case CharType(':'):
					OutToken = EJsonxToken::Colon;
					return true;

				case CharType(','):
					OutToken = EJsonxToken::Comma;
					return true;

				case CharType('\"'):
					{
						if (!ParseStringToken())
						{
							return false;
						}

						OutToken = EJsonxToken::String;
					}
					return true;

				case CharType('t'): case CharType('T'):
				case CharType('f'): case CharType('F'):
				case CharType('n'): case CharType('N'):
					{
						FString Test;
						Test += Char;

						while (!Stream->AtEnd())
						{
							if (!Serialize(&Char, sizeof(CharType)))
							{
								return false;
							}

							if (IsAlphaNumber(Char))
							{
								++CharacterNumber;
								Test += Char;
							}
							else
							{
								// backtrack and break
								Stream->Seek(Stream->Tell() - sizeof(CharType));
								break;
							}
						}

						if (Test == TEXT("False"))
						{
							BoolValue = false;
							OutToken = EJsonxToken::False;
							return true;
						}

						if (Test == TEXT("True"))
						{
							BoolValue = true;
							OutToken = EJsonxToken::True;
							return true;
						}

						if (Test == TEXT("Null"))
						{
							OutToken = EJsonxToken::Null;
							return true;
						}

						SetErrorMessage( TEXT("Invalid Jsonx Token. Check that your member names have quotes around them!") );
						return false;
					}

				default: 
					SetErrorMessage( TEXT("Invalid Jsonx Token.") );
					return false;
				}
			}
		}

		SetErrorMessage( TEXT("Invalid Jsonx Token.") );
		return false;
	}

	bool ParseStringToken()
	{
		FString String;

		while (true)
		{
			if (Stream->AtEnd())
			{
				SetErrorMessage( TEXT("String Token Abruptly Ended.") );
				return false;
			}

			CharType Char;
			if (!Serialize(&Char, sizeof(CharType)))
			{
				return false;
			}
			++CharacterNumber;

			if (Char == CharType('\"'))
			{
				break;
			}

			if (Char == CharType('\\'))
			{
				if (!Serialize(&Char, sizeof(CharType)))
				{
					return false;
				}
				++CharacterNumber;

				switch (Char)
				{
				case CharType('\"'): case CharType('\\'): case CharType('/'): String += Char; break;
				case CharType('f'): String += CharType('\f'); break;
				case CharType('r'): String += CharType('\r'); break;
				case CharType('n'): String += CharType('\n'); break;
				case CharType('b'): String += CharType('\b'); break;
				case CharType('t'): String += CharType('\t'); break;
				case CharType('u'):
					// 4 hex digits, like \uAB23, which is a 16 bit number that we would usually see as 0xAB23
					{
						int32 HexNum = 0;

						for (int32 Radix = 3; Radix >= 0; --Radix)
						{
							if (Stream->AtEnd())
							{
								SetErrorMessage( TEXT("String Token Abruptly Ended.") );
								return false;
							}

							if (!Serialize(&Char, sizeof(CharType)))
							{
								return false;
							}
							++CharacterNumber;

							int32 HexDigit = FParse::HexDigit(Char);

							if ((HexDigit == 0) && (Char != CharType('0')))
							{
								SetErrorMessage( TEXT("Invalid Hexadecimal digit parsed.") );
								return false;
							}

							//@TODO: FLOATPRECISION: this is gross
							HexNum += HexDigit * (int32)FMath::Pow(16, (float)Radix);
						}

						String += (FString::ElementType)HexNum;
					}
					break;

				default:
					SetErrorMessage( TEXT("Bad Jsonx escaped char.") );
					return false;
				}
			}
			else
			{
				String += Char;
			}
		}

		StringValue = MoveTemp(String);

		// Inline combine any surrogate pairs in the data when loading into a UTF-32 string
		StringConv::InlineCombineSurrogates(StringValue);

		return true;
	}

	bool ParseNumberToken( CharType FirstChar )
	{
		FString String;
		int32 State = 0;
		bool UseFirstChar = true;
		bool StateError = false;

		while (true)
		{
			if (Stream->AtEnd())
			{
				SetErrorMessage( TEXT("Number Token Abruptly Ended.") );
				return false;
			}

			CharType Char;
			if (UseFirstChar)
			{
				Char = FirstChar;
				UseFirstChar = false;
			}
			else
			{
				if (!Serialize(&Char, sizeof(CharType)))
				{
					return false;
				}
				++CharacterNumber;
			}

			// The following code doesn't actually derive the Jsonx Number: that is handled
			// by the function FCString::Atof below. This code only ensures the Jsonx Number is
			// EXACTLY to specification
			if (IsJsonxNumber(Char))
			{
				// ensure number follows Jsonx format before converting
				// This switch statement is derived from a finite state automata
				// derived from the Jsonx spec. A table was not used for simplicity.
				switch (State)
				{
				case 0:
					if (Char == CharType('-')) { State = 1; }
					else if (Char == CharType('0')) { State = 2; }
					else if (IsNonZeroDigit(Char)) { State = 3; }
					else { StateError = true; }
					break;

				case 1:
					if (Char == CharType('0')) { State = 2; }
					else if (IsNonZeroDigit(Char)) { State = 3; }
					else { StateError = true; }
					break;

				case 2:
					if (Char == CharType('.')) { State = 4; }
					else if (Char == CharType('e') || Char == CharType('E')) { State = 5; }
					else { StateError = true; }
					break;

				case 3:
					if (IsDigit(Char)) { State = 3; }
					else if (Char == CharType('.')) { State = 4; }
					else if (Char == CharType('e') || Char == CharType('E')) { State = 5; }
					else { StateError = true; }
					break;

				case 4:
					if (IsDigit(Char)) { State = 6; }
					else { StateError = true; }
					break;

				case 5:
					if (Char == CharType('-') ||Char == CharType('+')) { State = 7; }
					else if (IsDigit(Char)) { State = 8; }
					else { StateError = true; }
					break;

				case 6:
					if (IsDigit(Char)) { State = 6; }
					else if (Char == CharType('e') || Char == CharType('E')) { State = 5; }
					else { StateError = true; }
					break;

				case 7:
					if (IsDigit(Char)) { State = 8; }
					else { StateError = true; }
					break;

				case 8:
					if (IsDigit(Char)) { State = 8; }
					else { StateError = true; }
					break;

				default:
					SetErrorMessage( TEXT("Unknown state reached in Jsonx Number Token.") );
					return false;
				}

				if (StateError)
				{
					break;
				}

				String += Char;
			}
			else
			{
				// backtrack once because we read a non-number character
				Stream->Seek(Stream->Tell() - sizeof(CharType));
				--CharacterNumber;
				// and now the number is fully tokenized
				break;
			}
		}

		// ensure the number has followed valid Jsonx format
		if (!StateError && ((State == 2) || (State == 3) || (State == 6) || (State == 8)))
		{
			StringValue = String;
			NumberValue = FCString::Atod(*String);
			return true;
		}

		SetErrorMessage( TEXT("Poorly formed Jsonx Number Token.") );
		return false;
	}

	bool ParseWhiteSpace()
	{
		while (!Stream->AtEnd())
		{
			CharType Char;
			if (!Serialize(&Char, sizeof(CharType)))
			{
				return false;
			}
			++CharacterNumber;

			if (IsLineBreak(Char))
			{
				++LineNumber;
				CharacterNumber = 0;
			}

			if (!IsWhitespace(Char))
			{
				// backtrack and break
				Stream->Seek(Stream->Tell() - sizeof(CharType));
				--CharacterNumber;
				break;
			}
		}
		return true;
	}

	bool IsLineBreak( const CharType& Char )
	{
		return Char == CharType('\n');
	}

	/** Can't use FChar::IsWhitespace because it is TCHAR specific, and it doesn't handle newlines */
	bool IsWhitespace( const CharType& Char )
	{
		return Char == CharType(' ') || Char == CharType('\t') || Char == CharType('\n') || Char == CharType('\r');
	}

	/** Can't use FChar::IsDigit because it is TCHAR specific, and it doesn't handle all the other Jsonx number characters */
	bool IsJsonxNumber( const CharType& Char )
	{
		return ((Char >= CharType('0') && Char <= CharType('9')) ||
			Char == CharType('-') || Char == CharType('.') || Char == CharType('+') || Char == CharType('e') || Char == CharType('E'));
	}

	/** Can't use FChar::IsDigit because it is TCHAR specific */
	bool IsDigit( const CharType& Char )
	{
		return (Char >= CharType('0') && Char <= CharType('9'));
	}

	bool IsNonZeroDigit( const CharType& Char )
	{
		return (Char >= CharType('1') && Char <= CharType('9'));
	}

	/** Can't use FChar::IsAlpha because it is TCHAR specific. Also, this only checks A through Z (no underscores or other characters). */
	bool IsAlphaNumber( const CharType& Char )
	{
		return (Char >= CharType('a') && Char <= CharType('z')) || (Char >= CharType('A') && Char <= CharType('Z'));
	}

protected:
	bool Serialize(void* V, int64 Length)
	{
		Stream->Serialize(V, Length);
		if (Stream->IsError())
		{
			SetErrorMessage(TEXT("Stream I/O Error"));
			return false;
		}
		return true;
	}

protected:

	TArray<EJsonx> ParseState;
	EJsonxToken CurrentToken;

	FArchive* Stream;
	FString Identifier;
	FString ErrorMessage;
	FString StringValue;
	double NumberValue;
	uint32 LineNumber;
	uint32 CharacterNumber;
	bool BoolValue;
	bool FinishedReadingRootObject;
};


class FJsonxStringReader
	: public TJsonxReader<TCHAR>
{
public:

	static TSharedRef<FJsonxStringReader> Create(const FString& JsonxString)
	{
		return MakeShareable(new FJsonxStringReader(JsonxString));
	}

	static TSharedRef<FJsonxStringReader> Create(FString&& JsonxString)
	{
		return MakeShareable(new FJsonxStringReader(MoveTemp(JsonxString)));
	}

	const FString& GetSourceString() const
	{
		return Content;
	}
public:

	virtual ~FJsonxStringReader() = default;

protected:

	/**
	 * Parses a string containing Jsonx information.
	 *
	 * @param JsonxString The Jsonx string to parse.
	 */
	FJsonxStringReader(const FString& JsonxString)
		: Content(JsonxString)
		, Reader(nullptr)
	{
		InitReader();
	}

	/**
	 * Parses a string containing Jsonx information.
	 *
	 * @param JsonxString The Jsonx string to parse.
	 */
	FJsonxStringReader(FString&& JsonxString)
		: Content(MoveTemp(JsonxString))
		, Reader(nullptr)
	{
		InitReader();
	}

	FORCEINLINE void InitReader()
	{
		if (Content.IsEmpty())
		{
			return;
		}

		Reader = MakeUnique<FBufferReader>((void*)*Content, Content.Len() * sizeof(TCHAR), false);
		check(Reader.IsValid());

		Stream = Reader.Get();
	}

protected:
	const FString Content;
	TUniquePtr<FBufferReader> Reader;
};


template <class CharType = TCHAR>
class TJsonxReaderFactory
{
public:

	static TSharedRef<TJsonxReader<TCHAR>> Create(const FString& JsonxString)
	{
		return FJsonxStringReader::Create(JsonxString);
	}

	static TSharedRef<TJsonxReader<TCHAR>> Create(FString&& JsonxString)
	{
		return FJsonxStringReader::Create(MoveTemp(JsonxString));
	}

	static TSharedRef<TJsonxReader<CharType>> Create(FArchive* const Stream)
	{
		return TJsonxReader<CharType>::Create(Stream);
	}
};
