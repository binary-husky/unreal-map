// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Policies/PrettyJsonxPrintPolicy.h"
#include "Policies/CondensedJsonxPrintPolicy.h"
#include "Serialization/JsonxTypes.h"
#include "Serialization/JsonxReader.h"
#include "Serialization/JsonxSerializer.h"

/**
 * Macros used to generate a serialization function for a class derived from FJsonxSerializable
 */
#define BEGIN_JSONX_SERIALIZER \
	virtual void Serialize(FJsonxSerializerBase& Serializer, bool bFlatObject) override \
	{ \
		if (!bFlatObject) { Serializer.StartObject(); }

#define END_JSONX_SERIALIZER \
		if (!bFlatObject) { Serializer.EndObject(); } \
	}

#define JSONX_SERIALIZE(JsonxName, JsonxValue) \
		Serializer.Serialize(TEXT(JsonxName), JsonxValue)

#define JSONX_SERIALIZE_OPTIONAL(JsonxName, OptionalJsonxValue) \
		if (Serializer.IsLoading()) \
		{ \
			if (Serializer.GetObject()->HasField(TEXT(JsonxName))) \
			{ \
				Serializer.Serialize(TEXT(JsonxName), OptionalJsonxValue.Emplace()); \
			} \
		} \
		else \
		{ \
			if (OptionalJsonxValue.IsSet()) \
			{ \
				Serializer.Serialize(TEXT(JsonxName), OptionalJsonxValue.GetValue()); \
			} \
		}

#define JSONX_SERIALIZE_ARRAY(JsonxName, JsonxArray) \
		Serializer.SerializeArray(TEXT(JsonxName), JsonxArray)

#define JSONX_SERIALIZE_MAP(JsonxName, JsonxMap) \
		Serializer.SerializeMap(TEXT(JsonxName), JsonxMap)

#define JSONX_SERIALIZE_SIMPLECOPY(JsonxMap) \
		Serializer.SerializeSimpleMap(JsonxMap)

#define JSONX_SERIALIZE_MAP_SAFE(JsonxName, JsonxMap) \
		Serializer.SerializeMapSafe(TEXT(JsonxName), JsonxMap)

#define JSONX_SERIALIZE_SERIALIZABLE(JsonxName, JsonxValue) \
		JsonxValue.Serialize(Serializer, false)

#define JSONX_SERIALIZE_RAW_JSONX_STRING(JsonxName, JsonxValue) \
		if (Serializer.IsLoading()) \
		{ \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Object>(TEXT(JsonxName))) \
			{ \
				TSharedPtr<FJsonxObject> JsonxObject = Serializer.GetObject()->GetObjectField(TEXT(JsonxName)); \
				if (JsonxObject.IsValid()) \
				{ \
					auto Writer = TJsonxWriterFactory<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>::Create(&JsonxValue); \
					FJsonxSerializer::Serialize(JsonxObject.ToSharedRef(), Writer); \
				} \
			} \
			else \
			{ \
				JsonxValue = FString(); \
			} \
		} \
		else \
		{ \
			if (!JsonxValue.IsEmpty()) \
			{ \
				Serializer.WriteIdentifierPrefix(TEXT(JsonxName)); \
				Serializer.WriteRawJSONXValue(*JsonxValue); \
			} \
		}

#define JSONX_SERIALIZE_ARRAY_SERIALIZABLE(JsonxName, JsonxArray, ElementType) \
		if (Serializer.IsLoading()) \
		{ \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Array>(JsonxName)) \
			{ \
				for (auto It = Serializer.GetObject()->GetArrayField(JsonxName).CreateConstIterator(); It; ++It) \
				{ \
					ElementType* Obj = new(JsonxArray) ElementType(); \
					Obj->FromJsonx((*It)->AsObject()); \
				} \
			} \
		} \
		else \
		{ \
			Serializer.StartArray(JsonxName); \
			for (auto It = JsonxArray.CreateIterator(); It; ++It) \
			{ \
				It->Serialize(Serializer, false); \
			} \
			Serializer.EndArray(); \
		}

#define JSONX_SERIALIZE_OPTIONAL_ARRAY_SERIALIZABLE(JsonxName, OptionalJsonxArray, ElementType) \
		if (Serializer.IsLoading()) \
		{ \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Array>(JsonxName)) \
			{ \
				TArray<ElementType>& JsonxArray = OptionalJsonxArray.Emplace(); \
				for (auto It = Serializer.GetObject()->GetArrayField(JsonxName).CreateConstIterator(); It; ++It) \
				{ \
					ElementType* Obj = new(JsonxArray) ElementType(); \
					Obj->FromJsonx((*It)->AsObject()); \
				} \
			} \
		} \
		else \
		{ \
			if (OptionalJsonxArray.IsSet()) \
			{ \
				Serializer.StartArray(JsonxName); \
				for (auto It = OptionalJsonxArray->CreateIterator(); It; ++It) \
				{ \
					It->Serialize(Serializer, false); \
				} \
				Serializer.EndArray(); \
			} \
		}

#define JSONX_SERIALIZE_MAP_SERIALIZABLE(JsonxName, JsonxMap, ElementType) \
		if (Serializer.IsLoading()) \
		{ \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Object>(JsonxName)) \
			{ \
				TSharedPtr<FJsonxObject> JsonxObj = Serializer.GetObject()->GetObjectField(JsonxName); \
				for (auto MapIt = JsonxObj->Values.CreateConstIterator(); MapIt; ++MapIt) \
				{ \
					ElementType NewEntry; \
					NewEntry.FromJsonx(MapIt.Value()->AsObject()); \
					JsonxMap.Add(MapIt.Key(), NewEntry); \
				} \
			} \
		} \
		else \
		{ \
			Serializer.StartObject(JsonxName); \
			for (auto It = JsonxMap.CreateIterator(); It; ++It) \
			{ \
				Serializer.StartObject(It.Key()); \
				It.Value().Serialize(Serializer, true); \
				Serializer.EndObject(); \
			} \
			Serializer.EndObject(); \
		}

#define JSONX_SERIALIZE_OBJECT_SERIALIZABLE(JsonxName, JsonxSerializableObject) \
		/* Process the JsonxName field differently because it is an object */ \
		if (Serializer.IsLoading()) \
		{ \
			/* Read in the value from the JsonxName field */ \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Object>(JsonxName)) \
			{ \
				TSharedPtr<FJsonxObject> JsonxObj = Serializer.GetObject()->GetObjectField(JsonxName); \
				if (JsonxObj.IsValid()) \
				{ \
					(JsonxSerializableObject).FromJsonx(JsonxObj); \
				} \
			} \
		} \
		else \
		{ \
			/* Write the value to the Name field */ \
			Serializer.StartObject(JsonxName); \
			(JsonxSerializableObject).Serialize(Serializer, true); \
			Serializer.EndObject(); \
		}

#define JSONX_SERIALIZE_OPTIONAL_OBJECT_SERIALIZABLE(JsonxName, JsonxSerializableObject) \
		if (Serializer.IsLoading()) \
		{ \
			using ObjectType = TRemoveReference<decltype(JsonxSerializableObject.GetValue())>::Type; \
			if (Serializer.GetObject()->HasTypedField<EJsonx::Object>(JsonxName)) \
			{ \
				TSharedPtr<FJsonxObject> JsonxObj = Serializer.GetObject()->GetObjectField(JsonxName); \
				if (JsonxObj.IsValid()) \
				{ \
					JsonxSerializableObject = ObjectType{}; \
					JsonxSerializableObject.GetValue().FromJsonx(JsonxObj); \
				} \
			} \
		} \
		else \
		{ \
			if (JsonxSerializableObject.IsSet()) \
			{ \
				Serializer.StartObject(JsonxName); \
				(JsonxSerializableObject.GetValue()).Serialize(Serializer, true); \
				Serializer.EndObject(); \
			} \
		}

#define JSONX_SERIALIZE_DATETIME_UNIX_TIMESTAMP(JsonxName, JsonxDateTime) \
		if (Serializer.IsLoading()) \
		{ \
			int64 UnixTimestampValue; \
			Serializer.Serialize(TEXT(JsonxName), UnixTimestampValue); \
			JsonxDateTime = FDateTime::FromUnixTimestamp(UnixTimestampValue); \
		} \
		else \
		{ \
			int64 UnixTimestampValue = JsonxDateTime.ToUnixTimestamp(); \
			Serializer.Serialize(TEXT(JsonxName), UnixTimestampValue); \
		}

#define JSONX_SERIALIZE_DATETIME_UNIX_TIMESTAMP_MILLISECONDS(JsonxName, JsonxDateTime) \
if (Serializer.IsLoading()) \
{ \
	int64 UnixTimestampValueInMilliseconds; \
	Serializer.Serialize(TEXT(JsonxName), UnixTimestampValueInMilliseconds); \
	JsonxDateTime = FDateTime::FromUnixTimestamp(UnixTimestampValueInMilliseconds / 1000); \
} \
else \
{ \
	int64 UnixTimestampValueInMilliseconds = JsonxDateTime.ToUnixTimestamp() * 1000; \
	Serializer.Serialize(TEXT(JsonxName), UnixTimestampValueInMilliseconds); \
}

#define JSONX_SERIALIZE_ENUM(JsonxName, JsonxEnum) \
if (Serializer.IsLoading()) \
{ \
	FString JsonxTextValue; \
	Serializer.Serialize(TEXT(JsonxName), JsonxTextValue); \
	LexFromString(JsonxEnum, *JsonxTextValue); \
} \
else \
{ \
	FString JsonxTextValue = LexToString(JsonxEnum); \
	Serializer.Serialize(TEXT(JsonxName), JsonxTextValue); \
}


struct FJsonxSerializerBase;

/** Array of data */
typedef TArray<FString> FJsonxSerializableArray;
typedef TArray<int32> FJsonxSerializableArrayInt;

/** Maps a key to a value */
typedef TMap<FString, FString> FJsonxSerializableKeyValueMap;
typedef TMap<FString, int32> FJsonxSerializableKeyValueMapInt;
typedef TMap<FString, int64> FJsonxSerializableKeyValueMapInt64;
typedef TMap<FString, float> FJsonxSerializableKeyValueMapFloat;

/**
 * Base interface used to serialize to/from JSONX. Hides the fact there are separate read/write classes
 */
struct FJsonxSerializerBase
{
	virtual bool IsLoading() const = 0;
	virtual bool IsSaving() const = 0;
	virtual void StartObject() = 0;
	virtual void StartObject(const FString& Name) = 0;
	virtual void EndObject() = 0;
	virtual void StartArray() = 0;
	virtual void StartArray(const FString& Name) = 0;
	virtual void EndArray() = 0;
	virtual void Serialize(const TCHAR* Name, int32& Value) = 0;
	virtual void Serialize(const TCHAR* Name, uint32& Value) = 0;
	virtual void Serialize(const TCHAR* Name, int64& Value) = 0;
	virtual void Serialize(const TCHAR* Name, bool& Value) = 0;
	virtual void Serialize(const TCHAR* Name, FString& Value) = 0;
	virtual void Serialize(const TCHAR* Name, FText& Value) = 0;
	virtual void Serialize(const TCHAR* Name, float& Value) = 0;
	virtual void Serialize(const TCHAR* Name, double& Value) = 0;
	virtual void Serialize(const TCHAR* Name, FDateTime& Value) = 0;
	virtual void SerializeArray(FJsonxSerializableArray& Array) = 0;
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArray& Value) = 0;
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArrayInt& Value) = 0;
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map) = 0;
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt& Map) = 0;
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt64& Map) = 0;
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapFloat& Map) = 0;
	virtual void SerializeSimpleMap(FJsonxSerializableKeyValueMap& Map) = 0;
	virtual void SerializeMapSafe(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map) = 0;
	virtual TSharedPtr<FJsonxObject> GetObject() = 0;
	virtual void WriteIdentifierPrefix(const TCHAR* Name) = 0;
	virtual void WriteRawJSONXValue(const TCHAR* Value) = 0;
};

/**
 * Implements the abstract serializer interface hiding the underlying writer object
 */
template <class CharType = TCHAR, class PrintPolicy = TPrettyJsonxPrintPolicy<CharType> >
class FJsonxSerializerWriter :
	public FJsonxSerializerBase
{
	/** The object to write the JSONX output to */
	TSharedRef<TJsonxWriter<CharType, PrintPolicy> > JsonxWriter;

public:

	/**
	 * Initializes the writer object
	 *
	 * @param InJsonxWriter the object to write the JSONX output to
	 */
	FJsonxSerializerWriter(TSharedRef<TJsonxWriter<CharType, PrintPolicy> > InJsonxWriter) :
		JsonxWriter(InJsonxWriter)
	{
	}

	virtual ~FJsonxSerializerWriter()
	{
	}

	/** Is the JSONX being read from */
	virtual bool IsLoading() const override { return false; }
	/** Is the JSONX being written to */
	virtual bool IsSaving() const override { return true; }
	/** Access to the root object */
	virtual TSharedPtr<FJsonxObject> GetObject() override { return TSharedPtr<FJsonxObject>(); }

	/**
	 * Starts a new object "{"
	 */
	virtual void StartObject() override
	{
		JsonxWriter->WriteObjectStart();
	}

	/**
	 * Starts a new object "{"
	 */
	virtual void StartObject(const FString& Name) override
	{
		JsonxWriter->WriteObjectStart(Name);
	}
	/**
	 * Completes the definition of an object "}"
	 */
	virtual void EndObject() override
	{
		JsonxWriter->WriteObjectEnd();
	}

	virtual void StartArray() override
	{
		JsonxWriter->WriteArrayStart();
	}

	virtual void StartArray(const FString& Name) override
	{
		JsonxWriter->WriteArrayStart(Name);
	}

	virtual void EndArray() override
	{
		JsonxWriter->WriteArrayEnd();
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, int32& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, uint32& Value) override
	{
		JsonxWriter->WriteValue(Name, static_cast<int64>(Value));
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, int64& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, bool& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, FString& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, FText& Value) override
	{
		JsonxWriter->WriteValue(Name, Value.ToString());
	}
	/**
	 * Writes the field name and the corresponding value to the JSONX data
	 *
	 * @param Name the field name to write out
	 * @param Value the value to write out
	 */
	virtual void Serialize(const TCHAR* Name, float& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	* Writes the field name and the corresponding value to the JSONX data
	*
	* @param Name the field name to write out
	* @param Value the value to write out
	*/
	virtual void Serialize(const TCHAR* Name, double& Value) override
	{
		JsonxWriter->WriteValue(Name, Value);
	}
	/**
	* Writes the field name and the corresponding value to the JSONX data
	*
	* @param Name the field name to write out
	* @param Value the value to write out
	*/
	virtual void Serialize(const TCHAR* Name, FDateTime& Value) override
	{
		if (Value.GetTicks() > 0)
		{
			JsonxWriter->WriteValue(Name, Value.ToIso8601());
		}
	}
	/**
	 * Serializes an array of values
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(FJsonxSerializableArray& Array) override
	{
		JsonxWriter->WriteArrayStart();
		// Iterate all of values
		for (FJsonxSerializableArray::TIterator ArrayIt(Array); ArrayIt; ++ArrayIt)
		{
			JsonxWriter->WriteValue(*ArrayIt);
		}
		JsonxWriter->WriteArrayEnd();
	}
	/**
	 * Serializes an array of values with an identifier
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArray& Array) override
	{
		JsonxWriter->WriteArrayStart(Name);
		// Iterate all of values
		for (FJsonxSerializableArray::ElementType& Item :  Array)
		{
			JsonxWriter->WriteValue(Item);
		}
		JsonxWriter->WriteArrayEnd();
	}
	/**
	 * Serializes an array of values with an identifier
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArrayInt& Array) override
	{
		JsonxWriter->WriteArrayStart(Name);
		// Iterate all of values
		for (FJsonxSerializableArrayInt::ElementType& Item : Array)
		{
			JsonxWriter->WriteValue(Item);
		}
		JsonxWriter->WriteArrayEnd();
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map) override
	{
		JsonxWriter->WriteObjectStart(Name);
		// Iterate all of the keys and their values
		for (FJsonxSerializableKeyValueMap::ElementType& Pair : Map)
		{
			Serialize(*Pair.Key, Pair.Value);
		}
		JsonxWriter->WriteObjectEnd();
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt& Map) override
	{
		JsonxWriter->WriteObjectStart(Name);
		// Iterate all of the keys and their values
		for (FJsonxSerializableKeyValueMapInt::ElementType& Pair : Map)
		{
			Serialize(*Pair.Key, Pair.Value);
		}
		JsonxWriter->WriteObjectEnd();
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt64& Map) override
	{
		JsonxWriter->WriteObjectStart(Name);
		// Iterate all of the keys and their values
		for (FJsonxSerializableKeyValueMapInt64::ElementType& Pair : Map)
		{
			Serialize(*Pair.Key, Pair.Value);
		}
		JsonxWriter->WriteObjectEnd();
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapFloat& Map) override
	{
		JsonxWriter->WriteObjectStart(Name);
		// Iterate all of the keys and their values
		for (FJsonxSerializableKeyValueMapFloat::ElementType& Pair : Map)
		{
			Serialize(*Pair.Key, Pair.Value);
		}
		JsonxWriter->WriteObjectEnd();
	}

	virtual void SerializeSimpleMap(FJsonxSerializableKeyValueMap& Map) override
	{
		// writing does nothing here, this is meant to read in all data from a json object 
		// writing is explicitly handled per key/type
	}

	/**
	 * Serializes keys and values from an object into a map.
	 *
	 * @param Name Name of property to serialize
	 * @param Map The Map to copy String values from
	 */
	virtual void SerializeMapSafe(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map)
	{
		SerializeMap(Name, Map);
	}

	virtual void WriteIdentifierPrefix(const TCHAR* Name)
	{
		JsonxWriter->WriteIdentifierPrefix(Name);
	}

	virtual void WriteRawJSONXValue(const TCHAR* Value)
	{
		JsonxWriter->WriteRawJSONXValue(Value);
	}
};

/**
 * Implements the abstract serializer interface hiding the underlying reader object
 */
class FJsonxSerializerReader :
	public FJsonxSerializerBase
{
	/** The object that holds the parsed JSONX data */
	TSharedPtr<FJsonxObject> JsonxObject;

public:
	/**
	 * Inits the base JSONX object that is being read from
	 *
	 * @param InJsonxObject the JSONX object to serialize from
	 */
	FJsonxSerializerReader(TSharedPtr<FJsonxObject> InJsonxObject) :
		JsonxObject(InJsonxObject)
	{
	}

	virtual ~FJsonxSerializerReader()
	{
	}

	/** Is the JSONX being read from */
	virtual bool IsLoading() const override { return true; }
	/** Is the JSONX being written to */
	virtual bool IsSaving() const override { return false; }
	/** Access to the root Jsonx object being read */
	virtual TSharedPtr<FJsonxObject> GetObject() override { return JsonxObject; }

	/** Ignored */
	virtual void StartObject() override
	{
		// Empty on purpose
	}
	/** Ignored */
	virtual void StartObject(const FString& Name) override
	{
		// Empty on purpose
	}
	/** Ignored */
	virtual void EndObject() override
	{
		// Empty on purpose
	}
	/** Ignored */
	virtual void StartArray() override
	{
		// Empty on purpose
	}
	/** Ignored */
	virtual void StartArray(const FString& Name) override
	{
		// Empty on purpose
	}
	/** Ignored */
	virtual void EndArray() override
	{
		// Empty on purpose
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, int32& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Number>(Name))
		{
			JsonxObject->TryGetNumberField(Name, Value);
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, uint32& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Number>(Name))
		{
			JsonxObject->TryGetNumberField(Name, Value);
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, int64& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Number>(Name))
		{
			JsonxObject->TryGetNumberField(Name, Value);
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, bool& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Boolean>(Name))
		{
			Value = JsonxObject->GetBoolField(Name);
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, FString& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::String>(Name))
		{
			Value = JsonxObject->GetStringField(Name);
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, FText& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::String>(Name))
		{
			Value = FText::FromString(JsonxObject->GetStringField(Name));
		}
	}
	/**
	 * If the underlying json object has the field, it is read into the value
	 *
	 * @param Name the name of the field to read
	 * @param Value the out value to read the data into
	 */
	virtual void Serialize(const TCHAR* Name, float& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Number>(Name))
		{
			Value = (float)JsonxObject->GetNumberField(Name);
		}
	}
	/**
	* If the underlying json object has the field, it is read into the value
	*
	* @param Name the name of the field to read
	* @param Value the out value to read the data into
	*/
	virtual void Serialize(const TCHAR* Name, double& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Number>(Name))
		{
			Value = JsonxObject->GetNumberField(Name);
		}
	}
	/**
	* Writes the field name and the corresponding value to the JSONX data
	*
	* @param Name the field name to write out
	* @param Value the value to write out
	*/
	virtual void Serialize(const TCHAR* Name, FDateTime& Value) override
	{
		if (JsonxObject->HasTypedField<EJsonx::String>(Name))
		{
			FDateTime::ParseIso8601(*JsonxObject->GetStringField(Name), Value);
		}
	}
	/**
	 * Serializes an array of values
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(FJsonxSerializableArray& Array) override
	{
		// @todo - higher level serialization is expecting a Jsonx Object
		check(0 && TEXT("Not implemented"));
	}
	/**
	 * Serializes an array of values with an identifier
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArray& Array) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Array>(Name))
		{
			TArray< TSharedPtr<FJsonxValue> > JsonxArray = JsonxObject->GetArrayField(Name);
			// Iterate all of the keys and their values
			for (TSharedPtr<FJsonxValue>& Value : JsonxArray)
			{
				Array.Add(Value->AsString());
			}
		}
	}
	/**
	 * Serializes an array of values with an identifier
	 *
	 * @param Name the name of the property to serialize
	 * @param Array the array to serialize
	 */
	virtual void SerializeArray(const TCHAR* Name, FJsonxSerializableArrayInt& Array) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Array>(Name))
		{
			TArray< TSharedPtr<FJsonxValue> > JsonxArray = JsonxObject->GetArrayField(Name);
			// Iterate all of the keys and their values
			for (TSharedPtr<FJsonxValue>& Value : JsonxArray)
			{
				Array.Add(Value->AsNumber());
			}
		}
	}
	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Object>(Name))
		{
			TSharedPtr<FJsonxObject> JsonxMap = JsonxObject->GetObjectField(Name);
			// Iterate all of the keys and their values
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& Pair : JsonxMap->Values)
			{
				Map.Add(Pair.Key, Pair.Value->AsString());
			}
		}
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt& Map) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Object>(Name))
		{
			TSharedPtr<FJsonxObject> JsonxMap = JsonxObject->GetObjectField(Name);
			// Iterate all of the keys and their values
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& Pair : JsonxMap->Values)
			{
				const int32 Value = (int32)Pair.Value->AsNumber();
				Map.Add(Pair.Key, Value);
			}
		}
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapInt64& Map) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Object>(Name))
		{
			TSharedPtr<FJsonxObject> JsonxMap = JsonxObject->GetObjectField(Name);
			// Iterate all of the keys and their values
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& Pair : JsonxMap->Values)
			{
				const int64 Value = (int64)Pair.Value->AsNumber();
				Map.Add(Pair.Key, Value);
			}
		}
	}

	/**
	 * Serializes the keys & values for map
	 *
	 * @param Name the name of the property to serialize
	 * @param Map the map to serialize
	 */
	virtual void SerializeMap(const TCHAR* Name, FJsonxSerializableKeyValueMapFloat& Map) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Object>(Name))
		{
			TSharedPtr<FJsonxObject> JsonxMap = JsonxObject->GetObjectField(Name);
			// Iterate all of the keys and their values
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& Pair : JsonxMap->Values)
			{
				const float Value = (float)Pair.Value->AsNumber();
				Map.Add(Pair.Key, Value);
			}
		}
	}

	virtual void SerializeSimpleMap(FJsonxSerializableKeyValueMap& Map) override
	{
		// Iterate all of the keys and their values, only taking simple types (not array/object), all in string form
		for (auto KeyValueIt = JsonxObject->Values.CreateConstIterator(); KeyValueIt; ++KeyValueIt)
		{
			FString Value;
			if (KeyValueIt.Value()->TryGetString(Value))
			{
				Map.Add(KeyValueIt.Key(), MoveTemp(Value));
			}
		}
	}

	/**
	 * Deserializes keys and values from an object into a map, but only if the value is trivially convertable to string.
	 *
	 * @param Name Name of property to deserialize
	 * @param Map The Map to fill with String values found
	 */
	virtual void SerializeMapSafe(const TCHAR* Name, FJsonxSerializableKeyValueMap& Map) override
	{
		if (JsonxObject->HasTypedField<EJsonx::Object>(Name))
		{
			// Iterate all of the keys and their values, only taking simple types (not array/object), all in string form
			TSharedPtr<FJsonxObject> JsonxMap = JsonxObject->GetObjectField(Name);
			for (const TPair<FString, TSharedPtr<FJsonxValue>>& Pair : JsonxMap->Values)
			{
				FString Value;
				if (Pair.Value->TryGetString(Value))
				{
					Map.Add(Pair.Key, MoveTemp(Value));
				}
			}
		}
	}

	virtual void WriteIdentifierPrefix(const TCHAR* Name)
	{
		// Should never be called on a reader
		check(false);
	}

	virtual void WriteRawJSONXValue(const TCHAR* Value)
	{
		// Should never be called on a reader
		check(false);
	}
};

/**
 * Base class for a JSONX serializable object
 */
struct FJsonxSerializable
{
	/**
	 *	Virtualize destructor as we provide overridable functions
	 */
	virtual ~FJsonxSerializable() {}

	/**
	 * Used to allow serialization of a const ref
	 *
	 * @return the corresponding json string
	 */
	inline const FString ToJsonx(bool bPrettyPrint = true) const
	{
		// Strip away const, because we use a single method that can read/write which requires non-const semantics
		// Otherwise, we'd have to have 2 separate macros for declaring const to json and non-const from json
		return ((FJsonxSerializable*)this)->ToJsonx(bPrettyPrint);
	}
	/**
	 * Serializes this object to its JSONX string form
	 *
	 * @param bPrettyPrint - If true, will use the pretty json formatter
	 * @return the corresponding json string
	 */
	virtual const FString ToJsonx(bool bPrettyPrint=true)
	{
		FString JsonxStr;
		if (bPrettyPrint)
		{
			TSharedRef<TJsonxWriter<> > JsonxWriter = TJsonxWriterFactory<>::Create(&JsonxStr);
			FJsonxSerializerWriter<> Serializer(JsonxWriter);
			Serialize(Serializer, false);
			JsonxWriter->Close();
		}
		else
		{
			TSharedRef< TJsonxWriter< TCHAR, TCondensedJsonxPrintPolicy< TCHAR > > > JsonxWriter = TJsonxWriterFactory< TCHAR, TCondensedJsonxPrintPolicy< TCHAR > >::Create( &JsonxStr );
			FJsonxSerializerWriter<TCHAR, TCondensedJsonxPrintPolicy< TCHAR >> Serializer(JsonxWriter);
			Serialize(Serializer, false);
			JsonxWriter->Close();
		}
		return JsonxStr;
	}
	virtual void ToJsonx(TSharedRef<TJsonxWriter<> >& JsonxWriter, bool bFlatObject) const
	{
		FJsonxSerializerWriter<> Serializer(JsonxWriter);
		((FJsonxSerializable*)this)->Serialize(Serializer, bFlatObject);
	}
	virtual void ToJsonx(TSharedRef< TJsonxWriter< TCHAR, TCondensedJsonxPrintPolicy< TCHAR > > >& JsonxWriter, bool bFlatObject) const
	{
		FJsonxSerializerWriter<TCHAR, TCondensedJsonxPrintPolicy< TCHAR >> Serializer(JsonxWriter);
		((FJsonxSerializable*)this)->Serialize(Serializer, bFlatObject);
	}

	/**
	 * Serializes the contents of a JSONX string into this object
	 *
	 * @param Jsonx the JSONX data to serialize from
	 */
	virtual bool FromJsonx(const FString& Jsonx)
	{
		return FromJsonx(CopyTemp(Jsonx));
	}

	/**
	 * Serializes the contents of a JSONX string into this object
	 *
	 * @param Jsonx the JSONX data to serialize from
	 */
	virtual bool FromJsonx(FString&& Jsonx)
	{
		TSharedPtr<FJsonxObject> JsonxObject;
		TSharedRef<TJsonxReader<> > JsonxReader = TJsonxReaderFactory<>::Create(MoveTemp(Jsonx));
		if (FJsonxSerializer::Deserialize(JsonxReader,JsonxObject) &&
			JsonxObject.IsValid())
		{
			FJsonxSerializerReader Serializer(JsonxObject);
			Serialize(Serializer, false);
			return true;
		}
		return false;
	}

	virtual bool FromJsonx(TSharedPtr<FJsonxObject> JsonxObject)
	{
		if (JsonxObject.IsValid())
		{
			FJsonxSerializerReader Serializer(JsonxObject);
			Serialize(Serializer, false);
			return true;
		}
		return false;
	}

	/**
	 * Abstract method that needs to be supplied using the macros
	 *
	 * @param Serializer the object that will perform serialization in/out of JSONX
	 * @param bFlatObject if true then no object wrapper is used
	 */
	virtual void Serialize(FJsonxSerializerBase& Serializer, bool bFlatObject) = 0;
};

/**
 * Useful if you just want access to the underlying FJsonxObject (for cases where the schema is loose or an outer system will do further de/serialization)
 */
struct FJsonxDataBag
	: public FJsonxSerializable
{
	virtual void Serialize(FJsonxSerializerBase& Serializer, bool bFlatObject) override
	{
		if (Serializer.IsLoading())
		{
			// just grab a reference to the underlying JSONX object
			JsonxObject = Serializer.GetObject();
		}
		else
		{
			if (!bFlatObject)
			{
				Serializer.StartObject();
			}

			if (JsonxObject.IsValid())
			{
				for (const auto& It : JsonxObject->Values)
				{
					TSharedPtr<FJsonxValue> JsonxValue = It.Value;
					if (JsonxValue.IsValid())
					{
						switch (JsonxValue->Type)
						{
							case EJsonx::Boolean:
							{
								auto Value = JsonxValue->AsBool();
								Serializer.Serialize(*It.Key, Value);
								break;
							}
							case EJsonx::Number:
							{
								auto Value = JsonxValue->AsNumber();
								Serializer.Serialize(*It.Key, Value);
								break;
							}
							case EJsonx::String:
							{
								auto Value = JsonxValue->AsString();
								Serializer.Serialize(*It.Key, Value);
								break;
							}
							case EJsonx::Array:
							{
								// if we have an array, serialize to string and write raw
								FString JsonxStr;
								auto Writer = TJsonxWriterFactory<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>::Create(&JsonxStr);
								FJsonxSerializer::Serialize(JsonxValue->AsArray(), Writer);
								Serializer.WriteIdentifierPrefix(*It.Key);
								Serializer.WriteRawJSONXValue(*JsonxStr);
								break;
							}
							case EJsonx::Object:
							{
								// if we have an object, serialize to string and write raw
								FString JsonxStr;
								auto Writer = TJsonxWriterFactory<TCHAR, TCondensedJsonxPrintPolicy<TCHAR>>::Create(&JsonxStr);
								FJsonxSerializer::Serialize(JsonxValue->AsObject().ToSharedRef(), Writer);
								// too bad there's no JsonxObject serialization method on FJsonxSerializerBase directly :-/
								Serializer.WriteIdentifierPrefix(*It.Key);
								Serializer.WriteRawJSONXValue(*JsonxStr);
								break;
							}
						}
					}
				}
			}

			if (!bFlatObject)
			{
				Serializer.EndObject();
			}
		}
	}

	double GetDouble(const FString& Key) const
	{
		const auto Jsonx = GetField(Key);
		return Jsonx.IsValid() ? Jsonx->AsNumber() : 0.0;
	}

	FString GetString(const FString& Key) const
	{
		const auto Jsonx = GetField(Key);
		return Jsonx.IsValid() ? Jsonx->AsString() : FString();
	}

	bool GetBool(const FString& Key) const
	{
		const auto Jsonx = GetField(Key);
		return Jsonx.IsValid() ? Jsonx->AsBool() : false;
	}

	TSharedPtr<const FJsonxValue> GetField(const FString& Key) const
	{
		if (JsonxObject.IsValid())
		{
			return JsonxObject->TryGetField(Key);
		}
		return TSharedPtr<const FJsonxValue>();
	}

	template<typename JSONX_TYPE, typename Arg>
	void SetField(const FString& Key, Arg&& Value)
	{
		SetFieldJsonx(Key, MakeShared<JSONX_TYPE>(MoveTempIfPossible(Value)));
	}

	void SetFieldJsonx(const FString& Key, const TSharedPtr<FJsonxValue>& Value)
	{
		if (!JsonxObject.IsValid())
		{
			JsonxObject = MakeShared<FJsonxObject>();
		}
		JsonxObject->SetField(Key, Value);
	}

public:
	TSharedPtr<FJsonxObject> JsonxObject;
};
