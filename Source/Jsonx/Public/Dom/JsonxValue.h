// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Serialization/JsonxTypes.h"

class FJsonxObject;

/**
 * A Jsonx Value is a structure that can be any of the Jsonx Types.
 * It should never be used on its, only its derived types should be used.
 */
class JSONX_API FJsonxValue
{
public:

	/** Returns this value as a double, logging an error and returning zero if this is not an Jsonx Number */
	double AsNumber() const;

	/** Returns this value as a string, logging an error and returning an empty string if not possible */
	FString AsString() const;

	/** Returns this value as a boolean, logging an error and returning false if not possible */
	bool AsBool() const;

	/** Returns this value as an array, logging an error and returning an empty array reference if not possible */
	const TArray< TSharedPtr<FJsonxValue> >& AsArray() const;

	/** Returns this value as an object, throwing an error if this is not an Jsonx Object */
	virtual const TSharedPtr<FJsonxObject>& AsObject() const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(double& OutNumber) const { return false; }

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(float& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(int8& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(int16& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(int32& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(int64& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(uint8& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(uint16& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(uint32& OutNumber) const;

	/** Tries to convert this value to a number, returning false if not possible */
	virtual bool TryGetNumber(uint64& OutNumber) const;

	/** Tries to convert this value to a string, returning false if not possible */
	virtual bool TryGetString(FString& OutString) const { return false; }

	/** Tries to convert this value to a bool, returning false if not possible */
	virtual bool TryGetBool(bool& OutBool) const { return false; }

	/** Tries to convert this value to an array, returning false if not possible */
	virtual bool TryGetArray(const TArray< TSharedPtr<FJsonxValue> >*& OutArray) const { return false; }

	/** Tries to convert this value to an object, returning false if not possible */
	virtual bool TryGetObject(const TSharedPtr<FJsonxObject>*& Object) const { return false; }

	/** Returns true if this value is a 'null' */
	bool IsNull() const { return Type == EJsonx::Null || Type == EJsonx::None; }

	/** Get a field of the same type as the argument */
	void AsArgumentType(double                          & Value) { Value = AsNumber(); }
	void AsArgumentType(FString                         & Value) { Value = AsString(); }
	void AsArgumentType(bool                            & Value) { Value = AsBool  (); }
	void AsArgumentType(TArray< TSharedPtr<FJsonxValue> >& Value) { Value = AsArray (); }
	void AsArgumentType(TSharedPtr<FJsonxObject>         & Value) { Value = AsObject(); }

	EJsonx Type;

	static bool CompareEqual(const FJsonxValue& Lhs, const FJsonxValue& Rhs);

protected:

	FJsonxValue() : Type(EJsonx::None) {}
	virtual ~FJsonxValue() {}

	virtual FString GetType() const = 0;

	void ErrorMessage(const FString& InType) const;
};

inline bool operator==(const FJsonxValue& Lhs, const FJsonxValue& Rhs)
{
	return FJsonxValue::CompareEqual(Lhs, Rhs);
}

inline bool operator!=(const FJsonxValue& Lhs, const FJsonxValue& Rhs)
{
	return !FJsonxValue::CompareEqual(Lhs, Rhs);
}


/** A Jsonx String Value. */
class JSONX_API FJsonxValueString : public FJsonxValue
{
public:
	FJsonxValueString(const FString& InString) : Value(InString) {Type = EJsonx::String;}

	virtual bool TryGetString(FString& OutString) const override	{ OutString = Value; return true; }
	virtual bool TryGetNumber(double& OutDouble) const override		{ if (Value.IsNumeric()) { OutDouble = FCString::Atod(*Value); return true; } else { return false; } }
	virtual bool TryGetNumber(int32& OutValue) const override		{ LexFromString(OutValue, *Value); return true; }
	virtual bool TryGetNumber(uint32& OutValue) const override		{ LexFromString(OutValue, *Value); return true; }
	virtual bool TryGetNumber(int64& OutValue) const override		{ LexFromString(OutValue, *Value); return true; }
	virtual bool TryGetNumber(uint64& OutValue) const override		{ LexFromString(OutValue, *Value); return true; }
	virtual bool TryGetBool(bool& OutBool) const override			{ OutBool = Value.ToBool(); return true; }

	// Way to check if string value is empty without copying the string 
	bool IsEmpty() const { return Value.IsEmpty(); }

protected:
	FString Value;

	virtual FString GetType() const override {return TEXT("String");}
};


/** A Jsonx Number Value. */
class JSONX_API FJsonxValueNumber : public FJsonxValue
{
public:
	FJsonxValueNumber(double InNumber) : Value(InNumber) {Type = EJsonx::Number;}
	virtual bool TryGetNumber(double& OutNumber) const override		{ OutNumber = Value; return true; }
	virtual bool TryGetBool(bool& OutBool) const override			{ OutBool = (Value != 0.0); return true; }
	virtual bool TryGetString(FString& OutString) const override	{ OutString = FString::SanitizeFloat(Value, 0); return true; }
	
protected:

	double Value;

	virtual FString GetType() const override {return TEXT("Number");}
};


/** A Jsonx Number Value, stored internally as a string so as not to lose precision */
class JSONX_API FJsonxValueNumberString : public FJsonxValue
{
public:
	FJsonxValueNumberString(const FString& InString) : Value(InString) { Type = EJsonx::Number; }

	virtual bool TryGetString(FString& OutString) const override { OutString = Value; return true; }
	virtual bool TryGetNumber(double& OutDouble) const override { return LexTryParseString(OutDouble, *Value); }
	virtual bool TryGetNumber(float &OutDouble) const override { return LexTryParseString(OutDouble, *Value); }
	virtual bool TryGetNumber(int8& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(int16& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(int32& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(int64& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(uint8& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(uint16& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(uint32& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetNumber(uint64& OutValue) const override { return LexTryParseString(OutValue, *Value); }
	virtual bool TryGetBool(bool& OutBool) const override { OutBool = Value.ToBool(); return true; }

protected:
	FString Value;

	virtual FString GetType() const override { return TEXT("NumberString"); }
};


/** A Jsonx Boolean Value. */
class JSONX_API FJsonxValueBoolean : public FJsonxValue
{
public:
	FJsonxValueBoolean(bool InBool) : Value(InBool) {Type = EJsonx::Boolean;}
	virtual bool TryGetNumber(double& OutNumber) const override		{ OutNumber = Value ? 1 : 0; return true; }
	virtual bool TryGetBool(bool& OutBool) const override			{ OutBool = Value; return true; }
	virtual bool TryGetString(FString& OutString) const override	{ OutString = Value ? TEXT("true") : TEXT("false"); return true; }
	
protected:
	bool Value;

	virtual FString GetType() const override {return TEXT("Boolean");}
};


/** A Jsonx Array Value. */
class JSONX_API FJsonxValueArray : public FJsonxValue
{
public:
	FJsonxValueArray(const TArray< TSharedPtr<FJsonxValue> >& InArray) : Value(InArray) {Type = EJsonx::Array;}
	virtual bool TryGetArray(const TArray< TSharedPtr<FJsonxValue> >*& OutArray) const override	{ OutArray = &Value; return true; }
	
protected:
	TArray< TSharedPtr<FJsonxValue> > Value;

	virtual FString GetType() const override {return TEXT("Array");}
};


/** A Jsonx Object Value. */
class JSONX_API FJsonxValueObject : public FJsonxValue
{
public:
	FJsonxValueObject(TSharedPtr<FJsonxObject> InObject) : Value(InObject) {Type = EJsonx::Object;}
	virtual bool TryGetObject(const TSharedPtr<FJsonxObject>*& OutObject) const override			{ OutObject = &Value; return true; }
	
protected:
	TSharedPtr<FJsonxObject> Value;

	virtual FString GetType() const override {return TEXT("Object");}
};


/** A Jsonx Null Value. */
class JSONX_API FJsonxValueNull : public FJsonxValue
{
public:
	FJsonxValueNull() {Type = EJsonx::Null;}

protected:
	virtual FString GetType() const override {return TEXT("Null");}
};
