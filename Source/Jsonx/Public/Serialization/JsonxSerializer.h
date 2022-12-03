// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Serialization/JsonxTypes.h"
#include "Serialization/JsonxReader.h"
#include "Dom/JsonxValue.h"
#include "Dom/JsonxObject.h"
#include "Serialization/JsonxWriter.h"

class Error;

class FJsonxSerializer
{
public:

	enum class EFlags
	{
		None = 0,
		StoreNumbersAsStrings = 1,
	};

	template <class CharType>
	static bool Deserialize(const TSharedRef<TJsonxReader<CharType>>& Reader, TArray<TSharedPtr<FJsonxValue>>& OutArray, EFlags InOptions = EFlags::None)
	{
		return Deserialize(*Reader, OutArray, InOptions);
	}

	template <class CharType>
	static bool Deserialize(TJsonxReader<CharType>& Reader, TArray<TSharedPtr<FJsonxValue>>& OutArray, EFlags InOptions = EFlags::None)
	{
		StackState State;
		if (!Deserialize(Reader, /*OUT*/State, InOptions))
		{
			return false;
		}

		// Empty array is ok.
		if (State.Type != EJsonx::Array)
		{
			return false;
		}

		OutArray = State.Array;

		return true;
	}

	template <class CharType>
	static bool Deserialize(const TSharedRef<TJsonxReader<CharType>>& Reader, TSharedPtr<FJsonxObject>& OutObject, EFlags InOptions = EFlags::None)
	{
		return Deserialize(*Reader, OutObject, InOptions);
	}
	template <class CharType>
	static bool Deserialize(TJsonxReader<CharType>& Reader, TSharedPtr<FJsonxObject>& OutObject, EFlags InOptions = EFlags::None)
	{
		StackState State;
		if (!Deserialize(Reader, /*OUT*/State, InOptions))
		{
			return false;
		}

		if (!State.Object.IsValid())
		{
			return false;
		}

		OutObject = State.Object;

		return true;
	}

	template <class CharType>
	static bool Deserialize(const TSharedRef<TJsonxReader<CharType>>& Reader, TSharedPtr<FJsonxValue>& OutValue, EFlags InOptions = EFlags::None)
	{
		return Deserialize(*Reader, OutValue, InOptions);
	}

	template <class CharType>
	static bool Deserialize(TJsonxReader<CharType>& Reader, TSharedPtr<FJsonxValue>& OutValue, EFlags InOptions = EFlags::None)
	{
		StackState State;
		if (!Deserialize(Reader, /*OUT*/State, InOptions))
		{
			return false;
		}

		switch (State.Type)
		{
		case EJsonx::Object:
			if (!State.Object.IsValid())
			{
				return false;
			}
			OutValue = MakeShared<FJsonxValueObject>(State.Object);
			break;
		case EJsonx::Array:
			OutValue = MakeShared<FJsonxValueArray>(State.Array);
			break;
		default:
			// FIXME: would be nice to handle non-composite root values but StackState Deserialize just drops them on the floor
			return false;
		}
		return true;
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TArray<TSharedPtr<FJsonxValue>>& Array, const TSharedRef<TJsonxWriter<CharType, PrintPolicy>>& Writer, bool bCloseWriter = true)
	{
		return Serialize(Array, *Writer, bCloseWriter);
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TArray<TSharedPtr<FJsonxValue>>& Array, TJsonxWriter<CharType, PrintPolicy>& Writer, bool bCloseWriter = true )
	{
		const TSharedRef<FElement> StartingElement = MakeShared<FElement>(Array);
		return FJsonxSerializer::Serialize<CharType, PrintPolicy>(StartingElement, Writer, bCloseWriter);
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TSharedRef<FJsonxObject>& Object, const TSharedRef<TJsonxWriter<CharType, PrintPolicy>>& Writer, bool bCloseWriter = true )
	{
		return Serialize(Object, *Writer, bCloseWriter);
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TSharedRef<FJsonxObject>& Object, TJsonxWriter<CharType, PrintPolicy>& Writer, bool bCloseWriter = true)
	{
		const TSharedRef<FElement> StartingElement = MakeShared<FElement>(Object);
		return FJsonxSerializer::Serialize<CharType, PrintPolicy>(StartingElement, Writer, bCloseWriter);
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TSharedPtr<FJsonxValue>& Value, const FString& Identifier, const TSharedRef<TJsonxWriter<CharType, PrintPolicy>>& Writer, bool bCloseWriter = true)
	{
		return Serialize(Value, Identifier, *Writer, bCloseWriter);
	}
	template <class CharType, class PrintPolicy>
	static bool Serialize(const TSharedPtr<FJsonxValue>& Value, const FString& Identifier, TJsonxWriter<CharType, PrintPolicy>& Writer, bool bCloseWriter = true)
	{
		const TSharedRef<FElement> StartingElement = MakeShared<FElement>(Identifier, Value);
		return FJsonxSerializer::Serialize<CharType, PrintPolicy>(StartingElement, Writer, bCloseWriter);
	}

private:

	struct StackState
	{
		EJsonx Type;
		FString Identifier;
		TArray<TSharedPtr<FJsonxValue>> Array;
		TSharedPtr<FJsonxObject> Object;
	};

	struct FElement
	{
		FElement( const TSharedPtr<FJsonxValue>& InValue )
			: Identifier()
			, Value(InValue)
			, HasBeenProcessed(false)
		{ }

		FElement( const TSharedRef<FJsonxObject>& Object )
			: Identifier()
			, Value(MakeShared<FJsonxValueObject>(Object))
			, HasBeenProcessed( false )
		{ }

		FElement( const TArray<TSharedPtr<FJsonxValue>>& Array )
			: Identifier()
			, Value(MakeShared<FJsonxValueArray>(Array))
			, HasBeenProcessed(false)
		{ }

		FElement( const FString& InIdentifier, const TSharedPtr< FJsonxValue >& InValue )
			: Identifier( InIdentifier )
			, Value( InValue )
			, HasBeenProcessed( false )
		{

		}

		FString Identifier;
		TSharedPtr< FJsonxValue > Value;
		bool HasBeenProcessed;
	};

private:

	template <class CharType>
	static bool Deserialize(TJsonxReader<CharType>& Reader, StackState& OutStackState, EFlags InOptions)
	{
		TArray<TSharedRef<StackState>> ScopeStack; 
		TSharedPtr<StackState> CurrentState;

		TSharedPtr<FJsonxValue> NewValue;
		EJsonxNotation Notation;

		while (Reader.ReadNext(Notation))
		{
			FString Identifier = Reader.GetIdentifier();
			NewValue.Reset();

			switch( Notation )
			{
			case EJsonxNotation::ObjectStart:
				{
					if (CurrentState.IsValid())
					{
						ScopeStack.Push(CurrentState.ToSharedRef());
					}

					CurrentState = MakeShared<StackState>();
					CurrentState->Type = EJsonx::Object;
					CurrentState->Identifier = Identifier;
					CurrentState->Object = MakeShared<FJsonxObject>();
				}
				break;

			case EJsonxNotation::ObjectEnd:
				{
					if (ScopeStack.Num() > 0)
					{
						Identifier = CurrentState->Identifier;
						NewValue = MakeShared<FJsonxValueObject>(CurrentState->Object);
						CurrentState = ScopeStack.Pop();
					}
				}
				break;

			case EJsonxNotation::ArrayStart:
				{
					if (CurrentState.IsValid())
					{
						ScopeStack.Push(CurrentState.ToSharedRef());
					}

					CurrentState = MakeShared<StackState>();
					CurrentState->Type = EJsonx::Array;
					CurrentState->Identifier = Identifier;
				}
				break;

			case EJsonxNotation::ArrayEnd:
				{
					if (ScopeStack.Num() > 0)
					{
						Identifier = CurrentState->Identifier;
						NewValue = MakeShared<FJsonxValueArray>(CurrentState->Array);
						CurrentState = ScopeStack.Pop();
					}
				}
				break;

			case EJsonxNotation::Boolean:
				NewValue = MakeShared<FJsonxValueBoolean>(Reader.GetValueAsBoolean());
				break;

			case EJsonxNotation::String:
				NewValue = MakeShared<FJsonxValueString>(Reader.GetValueAsString());
				break;

			case EJsonxNotation::Number:
				if (EnumHasAnyFlags(InOptions, EFlags::StoreNumbersAsStrings))
				{
					NewValue = MakeShared<FJsonxValueNumberString>(Reader.GetValueAsNumberString());
				}
				else
				{
					NewValue = MakeShared<FJsonxValueNumber>(Reader.GetValueAsNumber());
				}
				break;

			case EJsonxNotation::Null:
				NewValue = MakeShared<FJsonxValueNull>();
				break;

			case EJsonxNotation::Error:
				return false;
				break;
			}

			if (NewValue.IsValid() && CurrentState.IsValid())
			{
				if (CurrentState->Type == EJsonx::Object)
				{
					CurrentState->Object->SetField(Identifier, NewValue);
				}
				else
				{
					CurrentState->Array.Add(NewValue);
				}
			}
		}

		if (!CurrentState.IsValid() || !Reader.GetErrorMessage().IsEmpty())
		{
			return false;
		}

		OutStackState = *CurrentState.Get();

		return true;
	}

	template <class CharType, class PrintPolicy>
	static bool Serialize(const TSharedRef<FElement>& StartingElement, TJsonxWriter<CharType, PrintPolicy>& Writer, bool bCloseWriter)
	{
		TArray<TSharedRef<FElement>> ElementStack;
		ElementStack.Push(StartingElement);

		while (ElementStack.Num() > 0)
		{
			TSharedRef<FElement> Element = ElementStack.Pop();
			check(Element->Value->Type != EJsonx::None);

			switch (Element->Value->Type)
			{
			case EJsonx::Number:	
				{
					if (Element->Identifier.IsEmpty())
					{
						Writer.WriteValue(Element->Value->AsNumber());
					}
					else
					{
						Writer.WriteValue(Element->Identifier, Element->Value->AsNumber());
					}
				}
				break;

			case EJsonx::Boolean:					
				{
					if (Element->Identifier.IsEmpty())
					{
						Writer.WriteValue(Element->Value->AsBool());
					}
					else
					{
						Writer.WriteValue(Element->Identifier, Element->Value->AsBool());
					}
				}
				break;

			case EJsonx::String:
				{
					if (Element->Identifier.IsEmpty())
					{
						Writer.WriteValue(Element->Value->AsString());
					}
					else
					{
						Writer.WriteValue(Element->Identifier, Element->Value->AsString());
					}
				}
				break;

			case EJsonx::Null:
				{
					if (Element->Identifier.IsEmpty())
					{
						Writer.WriteNull();
					}
					else
					{
						Writer.WriteNull(Element->Identifier);
					}
				}
				break;

			case EJsonx::Array:
				{
					if (Element->HasBeenProcessed)
					{
						Writer.WriteArrayEnd();
					}
					else
					{
						Element->HasBeenProcessed = true;
						ElementStack.Push(Element);

						if (Element->Identifier.IsEmpty())
						{
							Writer.WriteArrayStart();
						}
						else
						{
							Writer.WriteArrayStart(Element->Identifier);
						}

						TArray<TSharedPtr<FJsonxValue>> Values = Element->Value->AsArray();

						for (int Index = Values.Num() - 1; Index >= 0; --Index)
						{
							ElementStack.Push(MakeShared<FElement>(Values[Index]));
						}
					}
				}
				break;

			case EJsonx::Object:
				{
					if (Element->HasBeenProcessed)
					{
						Writer.WriteObjectEnd();
					}
					else
					{
						Element->HasBeenProcessed = true;
						ElementStack.Push(Element);

						if (Element->Identifier.IsEmpty())
						{
							Writer.WriteObjectStart();
						}
						else
						{
							Writer.WriteObjectStart(Element->Identifier);
						}

						TArray<FString> Keys; 
						TArray<TSharedPtr<FJsonxValue>> Values;
						TSharedPtr<FJsonxObject> ElementObject = Element->Value->AsObject();
						ElementObject->Values.GenerateKeyArray(Keys);
						ElementObject->Values.GenerateValueArray(Values);

						check(Keys.Num() == Values.Num());

						for (int Index = Values.Num() - 1; Index >= 0; --Index)
						{
							ElementStack.Push(MakeShared<FElement>(Keys[Index], Values[Index]));
						}
					}
				}
				break;

			default: 
				UE_LOG(LogJsonx, Fatal,TEXT("Could not print Jsonx Value, unrecognized type."));
			}
		}

		if (bCloseWriter)
		{
			return Writer.Close();
		}
		else
		{
			return true;
		}
	}
};
