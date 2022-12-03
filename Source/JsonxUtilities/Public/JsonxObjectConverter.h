// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Class.h"
#include "Serialization/JsonxTypes.h"
#include "Dom/JsonxObject.h"
#include "Serialization/JsonxReader.h"
#include "Serialization/JsonxSerializer.h"
#include "JsonxObjectWrapper.h"

/** Class that handles converting Jsonx objects to and from UStructs */
class JSONXUTILITIES_API FJsonxObjectConverter
{
public:

	/** FName case insensitivity can make the casing of UPROPERTIES unpredictable. Attempt to standardize output. */
	static FString StandardizeCase(const FString &StringIn);

	/** Parse an FText from a json object (assumed to be of the form where keys are culture codes and values are strings) */
	static bool GetTextFromObject(const TSharedRef<FJsonxObject>& Obj, FText& TextOut);

	/** Convert a Jsonx value to text (takes some hints from the value name) */
	static bool GetTextFromField(const FString& FieldName, const TSharedPtr<FJsonxValue>& FieldValue, FText& TextOut);

public: // UStruct -> JSONX

	/**
	 * Optional callback that will be run when exporting a single property to Jsonx.
	 * If this returns a valid value it will be inserted into the export chain.
	 * If this returns nullptr or is not bound, it will try generic type-specific export behavior before falling back to outputting ExportText as a string.
	 */
	DECLARE_DELEGATE_RetVal_TwoParams(TSharedPtr<FJsonxValue>, CustomExportCallback, FProperty* /* Property */, const void* /* Value */);

	/**
	 * Utility Export Callback for having object properties expanded to full Jsonx.
	 */
	UE_DEPRECATED(4.25, "ObjectJsonxCallback has been deprecated - please remove the usage of it from your project")
	static TSharedPtr<FJsonxValue> ObjectJsonxCallback(FProperty* Property , const void* Value);

	/**
	 * Templated version of UStructToJsonxObject to try and make most of the params. Also serves as an example use case
	 *
	 * @param InStruct The UStruct instance to read from
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 * @return FJsonxObject pointer. Invalid if an error occurred.
	 */
	template<typename InStructType>
	static TSharedPtr<FJsonxObject> UStructToJsonxObject(const InStructType& InStruct, int64 CheckFlags = 0, int64 SkipFlags = 0, const CustomExportCallback* ExportCb = nullptr)
	{
		TSharedRef<FJsonxObject> JsonxObject = MakeShared<FJsonxObject>();
		if (UStructToJsonxObject(InStructType::StaticStruct(), &InStruct, JsonxObject, CheckFlags, SkipFlags, ExportCb))
		{
			return JsonxObject;
		}
		return TSharedPtr<FJsonxObject>(); // something went wrong
	}

	/**
	 * Converts from a UStruct to a Jsonx Object, using exportText
	 *
	 * @param StructDefinition UStruct definition that is looked over for properties
	 * @param Struct The UStruct instance to copy out of
	 * @param JsonxObject Jsonx Object to be filled in with data from the ustruct
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 *
	 * @return False if any properties failed to write
	 */
	static bool UStructToJsonxObject(const UStruct* StructDefinition, const void* Struct, TSharedRef<FJsonxObject> OutJsonxObject, int64 CheckFlags = 0, int64 SkipFlags = 0, const CustomExportCallback* ExportCb = nullptr);

	/**
	 * Converts from a UStruct to a json string containing an object, using exportText
	 *
	 * @param StructDefinition UStruct definition that is looked over for properties
	 * @param Struct The UStruct instance to copy out of
	 * @param JsonxObject Jsonx Object to be filled in with data from the ustruct
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 * @param Indent How many tabs to add to the json serializer
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 * @param bPrettyPrint Option to use pretty print (e.g., adds line endings) or condensed print
	 *
	 * @return False if any properties failed to write
	 */
	static bool UStructToJsonxObjectString(const UStruct* StructDefinition, const void* Struct, FString& OutJsonxString, int64 CheckFlags = 0, int64 SkipFlags = 0, int32 Indent = 0, const CustomExportCallback* ExportCb = nullptr, bool bPrettyPrint = true);

	/**
	 * Templated version; Converts from a UStruct to a json string containing an object, using exportText
	 *
	 * @param Struct The UStruct instance to copy out of
	 * @param JsonxObject Jsonx Object to be filled in with data from the ustruct
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 * @param Indent How many tabs to add to the json serializer
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 * @param bPrettyPrint Option to use pretty print (e.g., adds line endings) or condensed print
	 *
	 * @return False if any properties failed to write
	 */
	template<typename InStructType>
	static bool UStructToJsonxObjectString(const InStructType& InStruct, FString& OutJsonxString, int64 CheckFlags = 0, int64 SkipFlags = 0, int32 Indent = 0, const CustomExportCallback* ExportCb = nullptr, bool bPrettyPrint = true)
	{
		return UStructToJsonxObjectString(InStructType::StaticStruct(), &InStruct, OutJsonxString, CheckFlags, SkipFlags, Indent, ExportCb, bPrettyPrint);
	}

	/**
	 * Wrapper to UStructToJsonxObjectString that allows a print policy to be specified.
	 */
	template<typename CharType, template<typename> class PrintPolicy>
	static bool UStructToFormattedJsonxObjectString(const UStruct* StructDefinition, const void* Struct, FString& OutJsonxString, int64 CheckFlags = 0, int64 SkipFlags = 0, int32 Indent = 0, const CustomExportCallback* ExportCb = nullptr)
	{
		TSharedRef<FJsonxObject> JsonxObject = MakeShareable(new FJsonxObject());
		if (UStructToJsonxObject(StructDefinition, Struct, JsonxObject, CheckFlags, SkipFlags, ExportCb))
		{
			TSharedRef<TJsonxWriter<CharType, PrintPolicy<CharType>>> JsonxWriter = TJsonxWriterFactory<CharType, PrintPolicy<CharType>>::Create(&OutJsonxString, Indent);

			if (FJsonxSerializer::Serialize(JsonxObject, JsonxWriter))
			{
				JsonxWriter->Close();
				return true;
			}
			else
			{
				UE_LOG(LogJsonx, Warning, TEXT("UStructToFormattedObjectString - Unable to write out json"));
				JsonxWriter->Close();
			}
		}

		return false;
	}

	/**
	 * Converts from a UStruct to a set of json attributes (possibly from within a JsonxObject)
	 *
	 * @param StructDefinition UStruct definition that is looked over for properties
	 * @param Struct The UStruct instance to copy out of
	 * @param JsonxAttributes Map of attributes to copy in to
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 *
	 * @return False if any properties failed to write
	 */
	static bool UStructToJsonxAttributes(const UStruct* StructDefinition, const void* Struct, TMap< FString, TSharedPtr<FJsonxValue> >& OutJsonxAttributes, int64 CheckFlags = 0, int64 SkipFlags = 0, const CustomExportCallback* ExportCb = nullptr);

	/* * Converts from a FProperty to a Jsonx Value using exportText
	 *
	 * @param Property			The property to export
	 * @param Value				Pointer to the value of the property
	 * @param CheckFlags		Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags			Skip properties that match any of these flags
	 * @param ExportCb Optional callback to override export behavior, if this returns null it will fallback to the default
	 * @param OuterProperty		If applicable, the Array/Set/Map Property that contains this property
	 *
	 * @return					The constructed JsonxValue from the property
	 */
	static TSharedPtr<FJsonxValue> UPropertyToJsonxValue(FProperty* Property, const void* Value, int64 CheckFlags = 0, int64 SkipFlags = 0, const CustomExportCallback* ExportCb = nullptr, FProperty* OuterProperty = nullptr);

public: // JSONX -> UStruct

	/**
	 * Converts from a Jsonx Object to a UStruct, using importText
	 *
	 * @param JsonxObject Jsonx Object to copy data out of
	 * @param StructDefinition UStruct definition that is looked over for properties
	 * @param Struct The UStruct instance to copy in to
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 *
	 * @return False if any properties matched but failed to deserialize
	 */
	static bool JsonxObjectToUStruct(const TSharedRef<FJsonxObject>& JsonxObject, const UStruct* StructDefinition, void* OutStruct, int64 CheckFlags = 0, int64 SkipFlags = 0);

	/**
	 * Templated version of JsonxObjectToUStruct
	 *
	 * @param JsonxObject Jsonx Object to copy data out of
	 * @param OutStruct The UStruct instance to copy in to
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 *
	 * @return False if any properties matched but failed to deserialize
	 */
	template<typename OutStructType>
	static bool JsonxObjectToUStruct(const TSharedRef<FJsonxObject>& JsonxObject, OutStructType* OutStruct, int64 CheckFlags = 0, int64 SkipFlags = 0)
	{
		return JsonxObjectToUStruct(JsonxObject, OutStructType::StaticStruct(), OutStruct, CheckFlags, SkipFlags);
	}

	/**
	 * Converts a set of json attributes (possibly from within a JsonxObject) to a UStruct, using importText
	 *
	 * @param JsonxAttributes Jsonx Object to copy data out of
	 * @param StructDefinition UStruct definition that is looked over for properties
	 * @param OutStruct The UStruct instance to copy in to
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 *
	 * @return False if any properties matched but failed to deserialize
	 */
	static bool JsonxAttributesToUStruct(const TMap< FString, TSharedPtr<FJsonxValue> >& JsonxAttributes, const UStruct* StructDefinition, void* OutStruct, int64 CheckFlags = 0, int64 SkipFlags = 0);

	/**
	 * Converts a single JsonxValue to the corresponding FProperty (this may recurse if the property is a UStruct for instance).
	 *
	 * @param JsonxValue The value to assign to this property
	 * @param Property The FProperty definition of the property we're setting.
	 * @param OutValue Pointer to the property instance to be modified.
	 * @param CheckFlags Only convert sub-properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip sub-properties that match any of these flags
	 *
	 * @return False if the property failed to serialize
	 */
	static bool JsonxValueToUProperty(const TSharedPtr<FJsonxValue>& JsonxValue, FProperty* Property, void* OutValue, int64 CheckFlags = 0, int64 SkipFlags = 0);

	/**
	 * Converts from a json string containing an object to a UStruct
	 *
	 * @param JsonxString String containing JSONX formatted data.
	 * @param OutStruct The UStruct instance to copy in to
	 * @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	 * @param SkipFlags Skip properties that match any of these flags
	 *
	 * @return False if any properties matched but failed to deserialize
	 */
	template<typename OutStructType>
	static bool JsonxObjectStringToUStruct(const FString& JsonxString, OutStructType* OutStruct, int64 CheckFlags = 0, int64 SkipFlags = 0)
	{
		TSharedPtr<FJsonxObject> JsonxObject;
		TSharedRef<TJsonxReader<> > JsonxReader = TJsonxReaderFactory<>::Create(JsonxString);
		if (!FJsonxSerializer::Deserialize(JsonxReader, JsonxObject) || !JsonxObject.IsValid())
		{
			UE_LOG(LogJsonx, Warning, TEXT("JsonxObjectStringToUStruct - Unable to parse json=[%s]"), *JsonxString);
			return false;
		}
		if (!FJsonxObjectConverter::JsonxObjectToUStruct(JsonxObject.ToSharedRef(), OutStruct, CheckFlags, SkipFlags))
		{
			UE_LOG(LogJsonx, Warning, TEXT("JsonxObjectStringToUStruct - Unable to deserialize. json=[%s]"), *JsonxString);
			return false;
		}
		return true;
	}

	/**
	* Converts from a json string containing an array to an array of UStructs
	*
	* @param JsonxString String containing JSONX formatted data.
	* @param OutStructArray The UStruct array to copy in to
	* @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	* @param SkipFlags Skip properties that match any of these flags.
	*
	* @return False if any properties matched but failed to deserialize.
	*/
	template<typename OutStructType>
	static bool JsonxArrayStringToUStruct(const FString& JsonxString, TArray<OutStructType>* OutStructArray, int64 CheckFlags = 0, int64 SkipFlags = 0)
	{
		TArray<TSharedPtr<FJsonxValue> > JsonxArray;
		TSharedRef<TJsonxReader<> > JsonxReader = TJsonxReaderFactory<>::Create(JsonxString);
		if (!FJsonxSerializer::Deserialize(JsonxReader, JsonxArray))
		{
			UE_LOG(LogJsonx, Warning, TEXT("JsonxArrayStringToUStruct - Unable to parse. json=[%s]"), *JsonxString);
			return false;
		}
		if (!JsonxArrayToUStruct(JsonxArray, OutStructArray, CheckFlags, SkipFlags))
		{
			UE_LOG(LogJsonx, Warning, TEXT("JsonxArrayStringToUStruct - Error parsing one of the elements. json=[%s]"), *JsonxString);
			return false;
		}
		return true;
	}

	/**
	* Converts from an array of json values to an array of UStructs.
	*
	* @param JsonxArray Array containing json values to convert.
	* @param OutStructArray The UStruct array to copy in to
	* @param CheckFlags Only convert properties that match at least one of these flags. If 0 check all properties.
	* @param SkipFlags Skip properties that match any of these flags.
	*
	* @return False if any of the matching elements are not an object, or if one of the matching elements could not be converted to the specified UStruct type.
	*/
	template<typename OutStructType>
	static bool JsonxArrayToUStruct(const TArray<TSharedPtr<FJsonxValue>>& JsonxArray, TArray<OutStructType>* OutStructArray, int64 CheckFlags = 0, int64 SkipFlags = 0)
	{
		OutStructArray->SetNum(JsonxArray.Num());
		for (int32 i = 0; i < JsonxArray.Num(); ++i)
		{
			const auto& Value = JsonxArray[i];
			if (Value->Type != EJsonx::Object)
			{
				UE_LOG(LogJsonx, Warning, TEXT("JsonxArrayToUStruct - Array element [%i] was not an object."), i);
				return false;
			}
			if (!FJsonxObjectConverter::JsonxObjectToUStruct(Value->AsObject().ToSharedRef(), OutStructType::StaticStruct(), &(*OutStructArray)[i], CheckFlags, SkipFlags))
			{
				UE_LOG(LogJsonx, Warning, TEXT("JsonxArrayToUStruct - Unable to convert element [%i]."), i);
				return false;
			}
		}
		return true;
	}

	/*
	* Parses text arguments from Jsonx into a map
	* @param JsonxObject Object to parse arguments from
	*/
	static FFormatNamedArguments ParseTextArgumentsFromJsonx(const TSharedPtr<const FJsonxObject>& JsonxObject);
};
