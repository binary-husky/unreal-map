// Copyright Epic Games, Inc. All Rights Reserved.

#include "Dom/JsonxValue.h"
#include "Dom/JsonxObject.h"




double FJsonxValue::AsNumber() const
{
	double Number = 0.0;

	if (!TryGetNumber(Number))
	{
		ErrorMessage(TEXT("Number"));
	}

	return Number;
}


FString FJsonxValue::AsString() const 
{
	FString String;

	if (!TryGetString(String))
	{
		ErrorMessage(TEXT("String"));
	}

	return String;
}


bool FJsonxValue::AsBool() const 
{
	bool Bool = false;

	if (!TryGetBool(Bool))
	{
		ErrorMessage(TEXT("Boolean")); 
	}

	return Bool;
}


const TArray< TSharedPtr<FJsonxValue> >& FJsonxValue::AsArray() const
{
	const TArray<TSharedPtr<FJsonxValue>>* Array = nullptr;

	if (!TryGetArray(Array))
	{
		static const TArray< TSharedPtr<FJsonxValue> > EmptyArray;
		Array = &EmptyArray;
		ErrorMessage(TEXT("Array"));
	}

	return *Array;
}


const TSharedPtr<FJsonxObject>& FJsonxValue::AsObject() const
{
	const TSharedPtr<FJsonxObject>* Object = nullptr;

	if (!TryGetObject(Object))
	{
		static const TSharedPtr<FJsonxObject> EmptyObject = MakeShared<FJsonxObject>();
		Object = &EmptyObject;
		ErrorMessage(TEXT("Object"));
	}

	return *Object;
}

// -----------------------------------

template <typename T>
bool TryConvertNumber(const FJsonxValue& InValue, T& OutNumber)
{
	double Double;

	if (InValue.TryGetNumber(Double) && (Double >= TNumericLimits<T>::Min()) && (Double <= static_cast<double>(TNumericLimits<T>::Max())))
	{
		OutNumber = static_cast<T>(FMath::RoundHalfFromZero(Double));

		return true;
	}

	return false;
}

// Need special handling for int64/uint64, due to overflow in the numeric limits.
// 2^63-1 and 2^64-1 cannot be exactly represented as a double, so TNumericLimits<>::Max() gets rounded up to exactly 2^63 or 2^64 by the compiler's implicit cast to double.
// This breaks the overflow check in TryConvertNumber. We use "<" rather than "<=" along with the exact power-of-two double literal to fix this.
template <> bool TryConvertNumber<uint64>(const FJsonxValue& InValue, uint64& OutNumber)
{
	double Double;
	if (InValue.TryGetNumber(Double) && Double >= 0.0 && Double < 18446744073709551616.0)
	{
		OutNumber = static_cast<uint64>(FMath::RoundHalfFromZero(Double));
		return true;
	}

	return false;
}

template <> bool TryConvertNumber<int64>(const FJsonxValue& InValue, int64& OutNumber)
{
	double Double;
	if (InValue.TryGetNumber(Double) && Double >= -9223372036854775808.0 && Double < 9223372036854775808.0)
	{
		OutNumber = static_cast<int64>(FMath::RoundHalfFromZero(Double));
		return true;
	}

	return false;
}

// -----------------------------------

bool FJsonxValue::TryGetNumber(float& OutNumber) const
{
	double Double;

	if (TryGetNumber(Double))
	{
		OutNumber = static_cast<float>(Double);
		return true;
	}

	return false;
}

bool FJsonxValue::TryGetNumber(uint8& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(uint16& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(uint32& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(uint64& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(int8& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(int16& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(int32& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

bool FJsonxValue::TryGetNumber(int64& OutNumber) const
{
	return TryConvertNumber(*this, OutNumber);
}

//static 
bool FJsonxValue::CompareEqual( const FJsonxValue& Lhs, const FJsonxValue& Rhs )
{
	if (Lhs.Type != Rhs.Type)
	{
		return false;
	}

	switch (Lhs.Type)
	{
	case EJsonx::None:
	case EJsonx::Null:
		return true;

	case EJsonx::String:
		return Lhs.AsString() == Rhs.AsString();

	case EJsonx::Number:
		return Lhs.AsNumber() == Rhs.AsNumber();

	case EJsonx::Boolean:
		return Lhs.AsBool() == Rhs.AsBool();

	case EJsonx::Array:
		{
			const TArray< TSharedPtr<FJsonxValue> >& LhsArray = Lhs.AsArray();
			const TArray< TSharedPtr<FJsonxValue> >& RhsArray = Rhs.AsArray();

			if (LhsArray.Num() != RhsArray.Num())
			{
				return false;
			}

			// compare each element
			for (int32 i = 0; i < LhsArray.Num(); ++i)
			{
				if (!CompareEqual(*LhsArray[i], *RhsArray[i]))
				{
					return false;
				}
			}
		}
		return true;

	case EJsonx::Object:
		{
			const TSharedPtr<FJsonxObject>& LhsObject = Lhs.AsObject();
			const TSharedPtr<FJsonxObject>& RhsObject = Rhs.AsObject();

			if (LhsObject.IsValid() != RhsObject.IsValid())
			{
				return false;
			}

			if (LhsObject.IsValid())
			{
				if (LhsObject->Values.Num() != RhsObject->Values.Num())
				{
					return false;
				}

				// compare each element
				for (const auto& It : LhsObject->Values)
				{
					const FString& Key = It.Key;
					const TSharedPtr<FJsonxValue>* RhsValue = RhsObject->Values.Find(Key);
					if (RhsValue == NULL)
					{
						// not found in both objects
						return false;
					}

					const TSharedPtr<FJsonxValue>& LhsValue = It.Value;

					if (LhsValue.IsValid() != RhsValue->IsValid())
					{
						return false;
					}

					if (LhsValue.IsValid())
					{
						if (!CompareEqual(*LhsValue.Get(), *RhsValue->Get()))
						{
							return false;
						}
					}
				}
			}
		}
		return true;

	default:
		return false;
	}
}

void FJsonxValue::ErrorMessage(const FString& InType) const
{
	UE_LOG(LogJsonx, Error, TEXT("Jsonx Value of type '%s' used as a '%s'."), *GetType(), *InType);
}
