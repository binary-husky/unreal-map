// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "JsonxGlobals.h"
#include "Serialization/JsonxTypes.h"
#include "Dom/JsonxValue.h"

/**
 * A Jsonx Object is a structure holding an unordered set of name/value pairs.
 * In a Jsonx file, it is represented by everything between curly braces {}.
 */
class JSONX_API FJsonxObject
{
public:

	TMap<FString, TSharedPtr<FJsonxValue>> Values;

	template<EJsonx JsonxType>
	TSharedPtr<FJsonxValue> GetField( const FString& FieldName ) const
	{
		const TSharedPtr<FJsonxValue>* Field = Values.Find(FieldName);
		if ( Field != nullptr && Field->IsValid() )
		{
			if (JsonxType == EJsonx::None || (*Field)->Type == JsonxType)
			{
				return (*Field);
			}
			else
			{
				UE_LOG(LogJsonx, Warning, TEXT("Field %s is of the wrong type."), *FieldName);
			}
		}
		else
		{
			UE_LOG(LogJsonx, Warning, TEXT("Field %s was not found."), *FieldName);
		}

		return MakeShared<FJsonxValueNull>();
	}

	/**
	 * Attempts to get the field with the specified name.
	 *
	 * @param FieldName The name of the field to get.
	 * @return A pointer to the field, or nullptr if the field doesn't exist.
	 */
	TSharedPtr<FJsonxValue> TryGetField( const FString& FieldName ) const
	{
		const TSharedPtr<FJsonxValue>* Field = Values.Find(FieldName);
		return (Field != nullptr && Field->IsValid()) ? *Field : TSharedPtr<FJsonxValue>();
	}

	/**
	 * Checks whether a field with the specified name exists in the object.
	 *
	 * @param FieldName The name of the field to check.
	 * @return true if the field exists, false otherwise.
	 */
	bool HasField( const FString& FieldName) const
	{
		const TSharedPtr<FJsonxValue>* Field = Values.Find(FieldName);
		if(Field && Field->IsValid())
		{
			return true;
		}

		return false;
	}
	
	/**
	 * Checks whether a field with the specified name and type exists in the object.
	 *
	 * @param JsonxType The type of the field to check.
	 * @param FieldName The name of the field to check.
	 * @return true if the field exists, false otherwise.
	 */
	template<EJsonx JsonxType>
	bool HasTypedField(const FString& FieldName) const
	{
		const TSharedPtr<FJsonxValue>* Field = Values.Find(FieldName);
		if(Field && Field->IsValid() && ((*Field)->Type == JsonxType))
		{
			return true;
		}

		return false;
	}

	/**
	 * Sets the value of the field with the specified name.
	 *
	 * @param FieldName The name of the field to set.
	 * @param Value The value to set.
	 */
	void SetField( const FString& FieldName, const TSharedPtr<FJsonxValue>& Value );

	/**
	 * Removes the field with the specified name.
	 *
	 * @param FieldName The name of the field to remove.
	 */
	void RemoveField(const FString& FieldName);

	/**
	 * Gets the field with the specified name as a number.
	 *
	 * Ensures that the field is present and is of type Jsonx number.
	 *
	 * @param FieldName The name of the field to get.
	 * @return The field's value as a number.
	 */
	double GetNumberField(const FString& FieldName) const;

	/**
	 * Gets a numeric field and casts to an int32
	 */
	FORCEINLINE int32 GetIntegerField(const FString& FieldName) const
	{
		return (int32)GetNumberField(FieldName);
	}

	/** Get the field named FieldName as a number. Returns false if it doesn't exist or cannot be converted. */
	bool TryGetNumberField(const FString& FieldName, double& OutNumber) const;

	/** Get the field named FieldName as a number, and makes sure it's within int32 range. Returns false if it doesn't exist or cannot be converted. */
	bool TryGetNumberField(const FString& FieldName, int32& OutNumber) const;

	/** Get the field named FieldName as a number, and makes sure it's within uint32 range. Returns false if it doesn't exist or cannot be converted.  */
	bool TryGetNumberField(const FString& FieldName, uint32& OutNumber) const;

	/** Get the field named FieldName as a number. Returns false if it doesn't exist or cannot be converted. */
	bool TryGetNumberField(const FString& FieldName, int64& OutNumber) const;

	/** Add a field named FieldName with Number as value */
	void SetNumberField( const FString& FieldName, double Number );

	/** Get the field named FieldName as a string. */
	FString GetStringField(const FString& FieldName) const;

	/** Get the field named FieldName as a string. Returns false if it doesn't exist or cannot be converted. */
	bool TryGetStringField(const FString& FieldName, FString& OutString) const;

	/** Get the field named FieldName as an array of strings. Returns false if it doesn't exist or any member cannot be converted. */
	bool TryGetStringArrayField(const FString& FieldName, TArray<FString>& OutArray) const;

	/** Get the field named FieldName as an array of enums. Returns false if it doesn't exist or any member is not a string. */
	template<typename TEnum>
	bool TryGetEnumArrayField(const FString& FieldName, TArray<TEnum>& OutArray) const
	{
		TArray<FString> Strings;
		if (!TryGetStringArrayField(FieldName, Strings))
		{
			return false;
		}

		OutArray.Empty();
		for (const FString& String : Strings)
		{
			TEnum Value;
			if (LexTryParseString(Value, *String))
			{
				OutArray.Add(Value);
			}
		}
		return true;
	}

	/** Add a field named FieldName with value of StringValue */
	void SetStringField( const FString& FieldName, const FString& StringValue );

	/**
	 * Gets the field with the specified name as a boolean.
	 *
	 * Ensures that the field is present and is of type Jsonx number.
	 *
	 * @param FieldName The name of the field to get.
	 * @return The field's value as a boolean.
	 */
	bool GetBoolField(const FString& FieldName) const;

	/** Get the field named FieldName as a string. Returns false if it doesn't exist or cannot be converted. */
	bool TryGetBoolField(const FString& FieldName, bool& OutBool) const;

	/** Set a boolean field named FieldName and value of InValue */
	void SetBoolField( const FString& FieldName, bool InValue );

	/** Get the field named FieldName as an array. */
	const TArray< TSharedPtr<FJsonxValue> >& GetArrayField(const FString& FieldName) const;

	/** Try to get the field named FieldName as an array, or return false if it's another type */
	bool TryGetArrayField(const FString& FieldName, const TArray< TSharedPtr<FJsonxValue> >*& OutArray) const;

	/** Set an array field named FieldName and value of Array */
	void SetArrayField( const FString& FieldName, const TArray< TSharedPtr<FJsonxValue> >& Array );

	/**
	 * Gets the field with the specified name as a Jsonx object.
	 *
	 * Ensures that the field is present and is of type Jsonx object.
	 *
	 * @param FieldName The name of the field to get.
	 * @return The field's value as a Jsonx object.
	 */
	const TSharedPtr<FJsonxObject>& GetObjectField(const FString& FieldName) const;

	/** Try to get the field named FieldName as an object, or return false if it's another type */
	bool TryGetObjectField(const FString& FieldName, const TSharedPtr<FJsonxObject>*& OutObject) const;

	/** Set an ObjectField named FieldName and value of JsonxObject */
	void SetObjectField( const FString& FieldName, const TSharedPtr<FJsonxObject>& JsonxObject );
};
