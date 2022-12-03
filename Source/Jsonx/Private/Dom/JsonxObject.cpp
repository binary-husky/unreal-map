// Copyright Epic Games, Inc. All Rights Reserved.

#include "Dom/JsonxObject.h"


void FJsonxObject::SetField( const FString& FieldName, const TSharedPtr<FJsonxValue>& Value )
{
	this->Values.Add(FieldName, Value);
}


void FJsonxObject::RemoveField( const FString& FieldName )
{
	this->Values.Remove(FieldName);
}


double FJsonxObject::GetNumberField( const FString& FieldName ) const
{
	return GetField<EJsonx::None>(FieldName)->AsNumber();
}


bool FJsonxObject::TryGetNumberField( const FString& FieldName, double& OutNumber ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetNumber(OutNumber);
}


bool FJsonxObject::TryGetNumberField( const FString& FieldName, int32& OutNumber ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetNumber(OutNumber);
}


bool FJsonxObject::TryGetNumberField( const FString& FieldName, uint32& OutNumber ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetNumber(OutNumber);
}

bool FJsonxObject::TryGetNumberField(const FString& FieldName, int64& OutNumber) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetNumber(OutNumber);
}

void FJsonxObject::SetNumberField( const FString& FieldName, double Number )
{
	this->Values.Add(FieldName, MakeShared<FJsonxValueNumber>(Number));
}


FString FJsonxObject::GetStringField( const FString& FieldName ) const
{
	return GetField<EJsonx::None>(FieldName)->AsString();
}


bool FJsonxObject::TryGetStringField( const FString& FieldName, FString& OutString ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetString(OutString);
}


bool FJsonxObject::TryGetStringArrayField( const FString& FieldName, TArray<FString>& OutArray ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);

	if (!Field.IsValid())
	{
		return false;
	}
	 
	const TArray< TSharedPtr<FJsonxValue> > *Array;

	if (!Field->TryGetArray(Array))
	{
		return false;
	}

	for (int Idx = 0; Idx < Array->Num(); Idx++)
	{
		FString Element;

		if (!(*Array)[Idx]->TryGetString(Element))
		{
			return false;
		}

		OutArray.Add(Element);
	}

	return true;
}


void FJsonxObject::SetStringField( const FString& FieldName, const FString& StringValue )
{
	this->Values.Add(FieldName, MakeShared<FJsonxValueString>(StringValue));
}


bool FJsonxObject::GetBoolField( const FString& FieldName ) const
{
	return GetField<EJsonx::None>(FieldName)->AsBool();
}


bool FJsonxObject::TryGetBoolField( const FString& FieldName, bool& OutBool ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetBool(OutBool);
}


void FJsonxObject::SetBoolField( const FString& FieldName, bool InValue )
{
	this->Values.Add(FieldName, MakeShared<FJsonxValueBoolean>(InValue));
}


const TArray<TSharedPtr<FJsonxValue>>& FJsonxObject::GetArrayField( const FString& FieldName ) const
{
	return GetField<EJsonx::Array>(FieldName)->AsArray();
}


bool FJsonxObject::TryGetArrayField(const FString& FieldName, const TArray< TSharedPtr<FJsonxValue> >*& OutArray) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetArray(OutArray);
}


void FJsonxObject::SetArrayField( const FString& FieldName, const TArray< TSharedPtr<FJsonxValue> >& Array )
{
	this->Values.Add(FieldName, MakeShared<FJsonxValueArray>(Array));
}


const TSharedPtr<FJsonxObject>& FJsonxObject::GetObjectField( const FString& FieldName ) const
{
	return GetField<EJsonx::Object>(FieldName)->AsObject();
}


bool FJsonxObject::TryGetObjectField( const FString& FieldName, const TSharedPtr<FJsonxObject>*& OutObject ) const
{
	TSharedPtr<FJsonxValue> Field = TryGetField(FieldName);
	return Field.IsValid() && Field->TryGetObject(OutObject);
}


void FJsonxObject::SetObjectField( const FString& FieldName, const TSharedPtr<FJsonxObject>& JsonxObject )
{
	if (JsonxObject.IsValid())
	{
		this->Values.Add(FieldName, MakeShared<FJsonxValueObject>(JsonxObject.ToSharedRef()));
	}
	else
	{
		this->Values.Add(FieldName, MakeShared<FJsonxValueNull>());
	}
}
