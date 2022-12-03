// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Dom/JsonxValue.h"
#include "Dom/JsonxObject.h"
#include "Serialization/JsonxSerializer.h"

#include "Templates/IsFloatingPoint.h"
#include "Templates/IsIntegral.h"
#include "Templates/EnableIf.h"
#include "Templates/Invoke.h"

/**
 * Helpers for creating TSharedPtr<FJsonxValue> JSONX trees
 *
 * Simple example:
 *
 *	FJsonxDomBuilder::FArray InnerArray;
 *	InnerArray.Add(7.f, TEXT("Hello"), true);
 *
 *	FJsonxDomBuilder::FObject Object;
 *	Object.Set(TEXT("Array"), InnerArray);
 *	Object.Set(TEXT("Number"), 13.f);
 *
 *	Object.AsJsonxValue();
 *
 * produces {"Array": [7., "Hello", true], "Number": 13.}
 */

class FJsonxDomBuilder
{
public:
	class FArray;

	class FObject
	{
	public:
		FObject()
			: Object(MakeShared<FJsonxObject>())
		{
		}

		TSharedRef<FJsonxValueObject> AsJsonxValue() const
		{
			return MakeShared<FJsonxValueObject>(Object);
		}

		TSharedRef<FJsonxObject> AsJsonxObject() const
		{
			return Object;
		}

		template <template <class> class TPrintPolicy = TPrettyJsonxPrintPolicy>
		FString ToString() const
		{
			FString Result;
			auto JsonxWriter = TJsonxWriterFactory<TCHAR, TPrintPolicy<TCHAR>>::Create(&Result);
			FJsonxSerializer::Serialize(Object, JsonxWriter);
			return Result;
		}

		int Num() const
		{
			return Object->Values.Num();
		}

		FObject& Set(const FString& Key, const FArray& Arr)            { Object->SetField(Key, Arr.AsJsonxValue());                            return *this; }
		FObject& Set(const FString& Key, const FObject& Obj)           { Object->SetField(Key, Obj.AsJsonxValue());                            return *this; }

		FObject& Set(const FString& Key, const FString& Str)           { Object->SetField(Key, MakeShared<FJsonxValueString>(Str));            return *this; }

		template <class FNumber>
		typename TEnableIf<!TIsSame<FNumber, bool>::Value && (TIsIntegral<FNumber>::Value || TIsFloatingPoint<FNumber>::Value), FObject&>::Type
			Set(const FString& Key, FNumber Number)                    { Object->SetField(Key, MakeShared<FJsonxValueNumber>(Number));         return *this; }

		template <class FBool>
		typename TEnableIf<TIsSame<FBool, bool>::Value, FObject&>::Type
			Set(const FString& Key, FBool Boolean)                     { Object->SetField(Key, MakeShared<FJsonxValueBoolean>(Boolean));       return *this; }

		FObject& Set(const FString& Key, TYPE_OF_NULLPTR)              { Object->SetField(Key, MakeShared<FJsonxValueNull>());                 return *this; }

		FObject& Set(const FString& Key, TSharedPtr<FJsonxValue> Value) { Object->SetField(Key, Value ? Value : MakeShared<FJsonxValueNull>()); return *this; }

		void CopyIf(const FJsonxObject& Src, TFunctionRef<bool (const FString&, const FJsonxValue&)> Pred)
		{
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& KV: Src.Values)
			{
				if (ensure(KV.Value) && Pred(KV.Key, *KV.Value))
				{
					Object->SetField(KV.Key, KV.Value);
				}
			}
		}

	private:
		TSharedRef<FJsonxObject> Object;
	};

	class FArray
	{
	public:
		TSharedRef<FJsonxValueArray> AsJsonxValue() const
		{
			return MakeShared<FJsonxValueArray>(Array);
		}

		template <template <class> class TPrintPolicy = TPrettyJsonxPrintPolicy>
		FString ToString() const
		{
			FString Result;
			auto JsonxWriter = TJsonxWriterFactory<TCHAR, TPrintPolicy<TCHAR>>::Create(&Result);
			FJsonxSerializer::Serialize(Array, JsonxWriter);
			return Result;
		}

		int Num() const
		{
			return Array.Num();
		}

		FArray& Add(const FArray& Arr)            { Array.Emplace(Arr.AsJsonxValue());                      return *this; }
		FArray& Add(const FObject& Obj)           { Array.Emplace(Obj.AsJsonxValue());                      return *this; }

		FArray& Add(const FString& Str)           { Array.Emplace(MakeShared<FJsonxValueString>(Str));      return *this; }

		template <class FNumber>
		typename TEnableIf<TIsIntegral<FNumber>::Value || TIsFloatingPoint<FNumber>::Value, FArray&>::Type
			Add(FNumber Number)                   { Array.Emplace(MakeShared<FJsonxValueNumber>(Number));   return *this; }

		FArray& Add(bool Boolean)                 { Array.Emplace(MakeShared<FJsonxValueBoolean>(Boolean)); return *this; }
		FArray& Add(TYPE_OF_NULLPTR)              { Array.Emplace(MakeShared<FJsonxValueNull>());           return *this; }

		FArray& Add(TSharedPtr<FJsonxValue> Value) { Array.Emplace(Value);                                  return *this; }

		/** Add multiple values */
		template <class... FValue>
		typename TEnableIf<(sizeof...(FValue) > 1), FArray&>::Type
			Add(FValue&&... Value)
		{
			// This should be implemented with a fold expression when our compilers support it
			int Temp[] = {0, (Add(Forward<FValue>(Value)), 0)...};
			(void)Temp;
			return *this;
		}

		void CopyIf(const TArray<TSharedPtr<FJsonxValue>>& Src, TFunctionRef<bool (const FJsonxValue&)> Pred)
		{
			for (const TSharedPtr<FJsonxValue>& Value: Src)
			{
				if (ensure(Value) && Pred(*Value))
				{
					Array.Emplace(Value);
				}
			}
		}
	private:
		TArray<TSharedPtr<FJsonxValue>> Array;
	};
};
