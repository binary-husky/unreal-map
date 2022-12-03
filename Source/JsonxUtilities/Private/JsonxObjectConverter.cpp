// Copyright Epic Games, Inc. All Rights Reserved.

#include "JsonxObjectConverter.h"
#include "Internationalization/Culture.h"
#include "UObject/ObjectMacros.h"
#include "UObject/Class.h"
#include "UObject/UnrealType.h"
#include "UObject/EnumProperty.h"
#include "UObject/TextProperty.h"
#include "UObject/PropertyPortFlags.h"
#include "UObject/Package.h"
#include "Policies/CondensedJsonxPrintPolicy.h"
#include "JsonxObjectWrapper.h"

FString FJsonxObjectConverter::StandardizeCase(const FString &StringIn)
{
	// this probably won't work for all cases, consider downcaseing the string fully
	FString FixedString = StringIn;
	FixedString[0] = FChar::ToLower(FixedString[0]); // our json classes/variable start lower case
	FixedString.ReplaceInline(TEXT("ID"), TEXT("Id"), ESearchCase::CaseSensitive); // Id is standard instead of ID, some of our fnames use ID
	return FixedString;
}


namespace
{
	const FString ObjectClassNameKey = "_ClassName";

/** Convert property to JSONX, assuming either the property is not an array or the value is an individual array element */
TSharedPtr<FJsonxValue> ConvertScalarFPropertyToJsonxValue(FProperty* Property, const void* Value, int64 CheckFlags, int64 SkipFlags, const FJsonxObjectConverter::CustomExportCallback* ExportCb, FProperty* OuterProperty)
{
	// See if there's a custom export callback first, so it can override default behavior
	if (ExportCb && ExportCb->IsBound())
	{
		TSharedPtr<FJsonxValue> CustomValue = ExportCb->Execute(Property, Value);
		if (CustomValue.IsValid())
		{
			return CustomValue;
		}
		// fall through to default cases
	}

	if (FEnumProperty* EnumProperty = CastField<FEnumProperty>(Property))
	{
		// export enums as strings
		UEnum* EnumDef = EnumProperty->GetEnum();
		FString StringValue = EnumDef->GetNameStringByValue(EnumProperty->GetUnderlyingProperty()->GetSignedIntPropertyValue(Value));
		return MakeShared<FJsonxValueString>(StringValue);
	}
	else if (FNumericProperty *NumericProperty = CastField<FNumericProperty>(Property))
	{
		// see if it's an enum
		UEnum* EnumDef = NumericProperty->GetIntPropertyEnum();
		if (EnumDef != NULL)
		{
			// export enums as strings
			FString StringValue = EnumDef->GetNameStringByValue(NumericProperty->GetSignedIntPropertyValue(Value));
			return MakeShared<FJsonxValueString>(StringValue);
		}

		// We want to export numbers as numbers
		if (NumericProperty->IsFloatingPoint())
		{
			return MakeShared<FJsonxValueNumber>(NumericProperty->GetFloatingPointPropertyValue(Value));
		}
		else if (NumericProperty->IsInteger())
		{
			return MakeShared<FJsonxValueNumber>(NumericProperty->GetSignedIntPropertyValue(Value));
		}

		// fall through to default
	}
	else if (FBoolProperty *BoolProperty = CastField<FBoolProperty>(Property))
	{
		// Export bools as bools
		return MakeShared<FJsonxValueBoolean>(BoolProperty->GetPropertyValue(Value));
	}
	else if (FStrProperty *StringProperty = CastField<FStrProperty>(Property))
	{
		return MakeShared<FJsonxValueString>(StringProperty->GetPropertyValue(Value));
	}
	else if (FTextProperty *TextProperty = CastField<FTextProperty>(Property))
	{
		return MakeShared<FJsonxValueString>(TextProperty->GetPropertyValue(Value).ToString());
	}
	else if (FArrayProperty *ArrayProperty = CastField<FArrayProperty>(Property))
	{
		TArray< TSharedPtr<FJsonxValue> > Out;
		FScriptArrayHelper Helper(ArrayProperty, Value);
		for (int32 i=0, n=Helper.Num(); i<n; ++i)
		{
			TSharedPtr<FJsonxValue> Elem = FJsonxObjectConverter::UPropertyToJsonxValue(ArrayProperty->Inner, Helper.GetRawPtr(i), CheckFlags & ( ~CPF_ParmFlags ), SkipFlags, ExportCb, ArrayProperty);
			if ( Elem.IsValid() )
			{
				// add to the array
				Out.Push(Elem);
			}
		}
		return MakeShared<FJsonxValueArray>(Out);
	}
	else if ( FSetProperty* SetProperty = CastField<FSetProperty>(Property) )
	{
		TArray< TSharedPtr<FJsonxValue> > Out;
		FScriptSetHelper Helper(SetProperty, Value);
		for ( int32 i=0, n=Helper.Num(); n; ++i )
		{
			if ( Helper.IsValidIndex(i) )
			{
				TSharedPtr<FJsonxValue> Elem = FJsonxObjectConverter::UPropertyToJsonxValue(SetProperty->ElementProp, Helper.GetElementPtr(i), CheckFlags & ( ~CPF_ParmFlags ), SkipFlags, ExportCb, SetProperty);
				if ( Elem.IsValid() )
				{
					// add to the array
					Out.Push(Elem);
				}

				--n;
			}
		}
		return MakeShared<FJsonxValueArray>(Out);
	}
	else if ( FMapProperty* MapProperty = CastField<FMapProperty>(Property) )
	{
		TSharedRef<FJsonxObject> Out = MakeShared<FJsonxObject>();

		FScriptMapHelper Helper(MapProperty, Value);
		for ( int32 i=0, n = Helper.Num(); n; ++i )
		{
			if ( Helper.IsValidIndex(i) )
			{
				TSharedPtr<FJsonxValue> KeyElement = FJsonxObjectConverter::UPropertyToJsonxValue(MapProperty->KeyProp, Helper.GetKeyPtr(i), CheckFlags & ( ~CPF_ParmFlags ), SkipFlags, ExportCb, MapProperty);
				TSharedPtr<FJsonxValue> ValueElement = FJsonxObjectConverter::UPropertyToJsonxValue(MapProperty->ValueProp, Helper.GetValuePtr(i), CheckFlags & ( ~CPF_ParmFlags ), SkipFlags, ExportCb, MapProperty);
				if ( KeyElement.IsValid() && ValueElement.IsValid() )
				{
					FString KeyString;
					if (!KeyElement->TryGetString(KeyString))
					{
						MapProperty->KeyProp->ExportTextItem(KeyString, Helper.GetKeyPtr(i), nullptr, nullptr, 0);
						if (KeyString.IsEmpty())
						{
							UE_LOG(LogJsonx, Error, TEXT("Unable to convert key to string for property %s."), *MapProperty->GetName())
							KeyString = FString::Printf(TEXT("Unparsed Key %d"), i);
						}
					}

					Out->SetField(KeyString, ValueElement);
				}

				--n;
			}
		}

		return MakeShared<FJsonxValueObject>(Out);
	}
	else if (FStructProperty *StructProperty = CastField<FStructProperty>(Property))
	{
		UScriptStruct::ICppStructOps* TheCppStructOps = StructProperty->Struct->GetCppStructOps();
		// Intentionally exclude the JSONX Object wrapper, which specifically needs to export JSONX in an object representation instead of a string
		if (StructProperty->Struct != FJsonxObjectWrapper::StaticStruct() && TheCppStructOps && TheCppStructOps->HasExportTextItem())
		{
			FString OutValueStr;
			TheCppStructOps->ExportTextItem(OutValueStr, Value, nullptr, nullptr, PPF_None, nullptr);
			return MakeShared<FJsonxValueString>(OutValueStr);
		}

		TSharedRef<FJsonxObject> Out = MakeShared<FJsonxObject>();
		if (FJsonxObjectConverter::UStructToJsonxObject(StructProperty->Struct, Value, Out, CheckFlags & (~CPF_ParmFlags), SkipFlags, ExportCb))
		{
			return MakeShared<FJsonxValueObject>(Out);
		}
	}
	else if (FObjectProperty* ObjectProperty = CastField<FObjectProperty>(Property))
	{
		// Instanced properties should be copied by value, while normal UObject* properties should output as asset references
		UObject* Object = ObjectProperty->GetObjectPropertyValue(Value);
		if (Object && (ObjectProperty->HasAnyPropertyFlags(CPF_PersistentInstance) || (OuterProperty && OuterProperty->HasAnyPropertyFlags(CPF_PersistentInstance))))
		{
			TSharedRef<FJsonxObject> Out = MakeShared<FJsonxObject>();

			Out->SetStringField(ObjectClassNameKey, Object->GetClass()->GetFName().ToString());
			if (FJsonxObjectConverter::UStructToJsonxObject(ObjectProperty->GetObjectPropertyValue(Value)->GetClass(), Object, Out, CheckFlags, SkipFlags, ExportCb))
			{
				TSharedRef<FJsonxValueObject> JsonxObject = MakeShared<FJsonxValueObject>(Out);
				JsonxObject->Type = EJsonx::Object;
				return JsonxObject;
			}
		}
		else
		{
			FString StringValue;
			Property->ExportTextItem(StringValue, Value, nullptr, nullptr, PPF_None);
			return MakeShared<FJsonxValueString>(StringValue);
		}
	}
	else
	{
		// Default to export as string for everything else
		FString StringValue;
		Property->ExportTextItem(StringValue, Value, NULL, NULL, PPF_None);
		return MakeShared<FJsonxValueString>(StringValue);
	}

	// invalid
	return TSharedPtr<FJsonxValue>();
}
}

PRAGMA_DISABLE_DEPRECATION_WARNINGS

TSharedPtr<FJsonxValue> FJsonxObjectConverter::ObjectJsonxCallback(FProperty* Property, const void* Value)
{
	if (FObjectProperty* ObjectProperty = CastField<FObjectProperty>(Property))
	{
		if (!ObjectProperty->HasAnyFlags(RF_Transient)) // We are taking Transient to mean we don't want to serialize to Jsonx either (could make a new flag if nessasary)
		{
			TSharedRef<FJsonxObject> Out = MakeShared<FJsonxObject>();

			CustomExportCallback CustomCB;
			CustomCB.BindStatic(FJsonxObjectConverter::ObjectJsonxCallback);

			void** PtrToValuePtr = (void**)Value;

			if (FJsonxObjectConverter::UStructToJsonxObject(ObjectProperty->PropertyClass, (*PtrToValuePtr), Out, 0, 0, &CustomCB))
			{
				return MakeShared<FJsonxValueObject>(Out);
			}
		}
	}

	// invalid
	return TSharedPtr<FJsonxValue>();
}

PRAGMA_ENABLE_DEPRECATION_WARNINGS

TSharedPtr<FJsonxValue> FJsonxObjectConverter::UPropertyToJsonxValue(FProperty* Property, const void* Value, int64 CheckFlags, int64 SkipFlags, const CustomExportCallback* ExportCb, FProperty* OuterProperty)
{
	if (Property->ArrayDim == 1)
	{
		return ConvertScalarFPropertyToJsonxValue(Property, Value, CheckFlags, SkipFlags, ExportCb, OuterProperty);
	}

	TArray< TSharedPtr<FJsonxValue> > Array;
	for (int Index = 0; Index != Property->ArrayDim; ++Index)
	{
		Array.Add(ConvertScalarFPropertyToJsonxValue(Property, (char*)Value + Index * Property->ElementSize, CheckFlags, SkipFlags, ExportCb, OuterProperty));
	}
	return MakeShared<FJsonxValueArray>(Array);
}

bool FJsonxObjectConverter::UStructToJsonxObject(const UStruct* StructDefinition, const void* Struct, TSharedRef<FJsonxObject> OutJsonxObject, int64 CheckFlags, int64 SkipFlags, const CustomExportCallback* ExportCb)
{
	return UStructToJsonxAttributes(StructDefinition, Struct, OutJsonxObject->Values, CheckFlags, SkipFlags, ExportCb);
}

bool FJsonxObjectConverter::UStructToJsonxAttributes(const UStruct* StructDefinition, const void* Struct, TMap< FString, TSharedPtr<FJsonxValue> >& OutJsonxAttributes, int64 CheckFlags, int64 SkipFlags, const CustomExportCallback* ExportCb)
{
	if (SkipFlags == 0)
	{
		// If we have no specified skip flags, skip deprecated, transient and skip serialization by default when writing
		SkipFlags |= CPF_Deprecated | CPF_Transient;
	}

	if (StructDefinition == FJsonxObjectWrapper::StaticStruct())
	{
		// Just copy it into the object
		const FJsonxObjectWrapper* ProxyObject = (const FJsonxObjectWrapper *)Struct;

		if (ProxyObject->JsonxObject.IsValid())
		{
			OutJsonxAttributes = ProxyObject->JsonxObject->Values;
		}
		return true;
	}

	for (TFieldIterator<FProperty> It(StructDefinition); It; ++It)
	{
		FProperty* Property = *It;

		// Check to see if we should ignore this property
		if (CheckFlags != 0 && !Property->HasAnyPropertyFlags(CheckFlags))
		{
			continue;
		}
		if (Property->HasAnyPropertyFlags(SkipFlags))
		{
			continue;
		}

		FString VariableName = StandardizeCase(Property->GetName());
		const void* Value = Property->ContainerPtrToValuePtr<uint8>(Struct);

		// convert the property to a FJsonxValue
		TSharedPtr<FJsonxValue> JsonxValue = UPropertyToJsonxValue(Property, Value, CheckFlags, SkipFlags, ExportCb);
		if (!JsonxValue.IsValid())
		{
			FFieldClass* PropClass = Property->GetClass();
			UE_LOG(LogJsonx, Error, TEXT("UStructToJsonxObject - Unhandled property type '%s': %s"), *PropClass->GetName(), *Property->GetPathName());
			return false;
		}

		// set the value on the output object
		OutJsonxAttributes.Add(VariableName, JsonxValue);
	}

	return true;
}

template<class CharType, class PrintPolicy>
bool UStructToJsonxObjectStringInternal(const TSharedRef<FJsonxObject>& JsonxObject, FString& OutJsonxString, int32 Indent)
{
	TSharedRef<TJsonxWriter<CharType, PrintPolicy> > JsonxWriter = TJsonxWriterFactory<CharType, PrintPolicy>::Create(&OutJsonxString, Indent);
	bool bSuccess = FJsonxSerializer::Serialize(JsonxObject, JsonxWriter);
	JsonxWriter->Close();
	return bSuccess;
}

bool FJsonxObjectConverter::UStructToJsonxObjectString(const UStruct* StructDefinition, const void* Struct, FString& OutJsonxString, int64 CheckFlags, int64 SkipFlags, int32 Indent, const CustomExportCallback* ExportCb, bool bPrettyPrint)
{
	TSharedRef<FJsonxObject> JsonxObject = MakeShared<FJsonxObject>();
	if (UStructToJsonxObject(StructDefinition, Struct, JsonxObject, CheckFlags, SkipFlags, ExportCb))
	{
		bool bSuccess = false;
		if (bPrettyPrint)
		{
			bSuccess = UStructToJsonxObjectStringInternal<TCHAR, TPrettyJsonxPrintPolicy<TCHAR> >(JsonxObject, OutJsonxString, Indent);
		}
		else
		{
			bSuccess = UStructToJsonxObjectStringInternal<TCHAR, TCondensedJsonxPrintPolicy<TCHAR> >(JsonxObject, OutJsonxString, Indent);
		}
		if (bSuccess)
		{
			return true;
		}
		else
		{
			UE_LOG(LogJsonx, Warning, TEXT("UStructToJsonxObjectString - Unable to write out json"));
		}
	}

	return false;
}

//static
bool FJsonxObjectConverter::GetTextFromObject(const TSharedRef<FJsonxObject>& Obj, FText& TextOut)
{
	// get the prioritized culture name list
	FCultureRef CurrentCulture = FInternationalization::Get().GetCurrentCulture();
	TArray<FString> CultureList = CurrentCulture->GetPrioritizedParentCultureNames();

	// try to follow the fall back chain that the engine uses
	FString TextString;
	for (const FString& CultureCode : CultureList)
	{
		if (Obj->TryGetStringField(CultureCode, TextString))
		{
			TextOut = FText::FromString(TextString);
			return true;
		}
	}

	// try again but only search on the locale region (in the localized data). This is a common omission (i.e. en-US source text should be used if no en is defined)
	for (const FString& LocaleToMatch : CultureList)
	{
		int32 SeparatorPos;
		// only consider base language entries in culture chain (i.e. "en")
		if (!LocaleToMatch.FindChar('-', SeparatorPos))
		{
			for (const auto& Pair : Obj->Values)
			{
				// only consider coupled entries now (base ones would have been matched on first path) (i.e. "en-US")
				if (Pair.Key.FindChar('-', SeparatorPos))
				{
					if (Pair.Key.StartsWith(LocaleToMatch))
					{
						TextOut = FText::FromString(Pair.Value->AsString());
						return true;
					}
				}
			}
		}
	}

	// no luck, is this possibly an unrelated json object?
	return false;
}


namespace
{
	bool JsonxValueToFPropertyWithContainer(const TSharedPtr<FJsonxValue>& JsonxValue, FProperty* Property, void* OutValue, const UStruct* ContainerStruct, void* Container, int64 CheckFlags, int64 SkipFlags);
	bool JsonxAttributesToUStructWithContainer(const TMap< FString, TSharedPtr<FJsonxValue> >& JsonxAttributes, const UStruct* StructDefinition, void* OutStruct, const UStruct* ContainerStruct, void* Container, int64 CheckFlags, int64 SkipFlags);

	/** Convert JSONX to property, assuming either the property is not an array or the value is an individual array element */
	bool ConvertScalarJsonxValueToFPropertyWithContainer(const TSharedPtr<FJsonxValue>& JsonxValue, FProperty* Property, void* OutValue, const UStruct* ContainerStruct, void* Container, int64 CheckFlags, int64 SkipFlags)
	{
	if (FEnumProperty* EnumProperty = CastField<FEnumProperty>(Property))
	{
		if (JsonxValue->Type == EJsonx::String)
		{
			// see if we were passed a string for the enum
			const UEnum* Enum = EnumProperty->GetEnum();
			check(Enum);
			FString StrValue = JsonxValue->AsString();
			int64 IntValue = Enum->GetValueByName(FName(*StrValue));
			if (IntValue == INDEX_NONE)
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable import enum %s from string value %s for property %s"), *Enum->CppType, *StrValue, *Property->GetNameCPP());
				return false;
			}
			EnumProperty->GetUnderlyingProperty()->SetIntPropertyValue(OutValue, IntValue);
		}
		else
		{
			// AsNumber will log an error for completely inappropriate types (then give us a default)
			EnumProperty->GetUnderlyingProperty()->SetIntPropertyValue(OutValue, (int64)JsonxValue->AsNumber());
		}
	}
	else if (FNumericProperty *NumericProperty = CastField<FNumericProperty>(Property))
	{
		if (NumericProperty->IsEnum() && JsonxValue->Type == EJsonx::String)
		{
			// see if we were passed a string for the enum
			const UEnum* Enum = NumericProperty->GetIntPropertyEnum();
			check(Enum); // should be assured by IsEnum()
			FString StrValue = JsonxValue->AsString();
			int64 IntValue = Enum->GetValueByName(FName(*StrValue));
			if (IntValue == INDEX_NONE)
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable import enum %s from string value %s for property %s"), *Enum->CppType, *StrValue, *Property->GetNameCPP());
				return false;
			}
			NumericProperty->SetIntPropertyValue(OutValue, IntValue);
		}
		else if (NumericProperty->IsFloatingPoint())
		{
			// AsNumber will log an error for completely inappropriate types (then give us a default)
			NumericProperty->SetFloatingPointPropertyValue(OutValue, JsonxValue->AsNumber());
		}
		else if (NumericProperty->IsInteger())
		{
			if (JsonxValue->Type == EJsonx::String)
			{
				// parse string -> int64 ourselves so we don't lose any precision going through AsNumber (aka double)
				NumericProperty->SetIntPropertyValue(OutValue, FCString::Atoi64(*JsonxValue->AsString()));
			}
			else
			{
				// AsNumber will log an error for completely inappropriate types (then give us a default)
				NumericProperty->SetIntPropertyValue(OutValue, (int64)JsonxValue->AsNumber());
			}
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable to set numeric property type %s for property %s"), *Property->GetClass()->GetName(), *Property->GetNameCPP());
			return false;
		}
	}
	else if (FBoolProperty *BoolProperty = CastField<FBoolProperty>(Property))
	{
		// AsBool will log an error for completely inappropriate types (then give us a default)
		BoolProperty->SetPropertyValue(OutValue, JsonxValue->AsBool());
	}
	else if (FStrProperty *StringProperty = CastField<FStrProperty>(Property))
	{
		// AsString will log an error for completely inappropriate types (then give us a default)
		StringProperty->SetPropertyValue(OutValue, JsonxValue->AsString());
	}
	else if (FArrayProperty *ArrayProperty = CastField<FArrayProperty>(Property))
	{
		if (JsonxValue->Type == EJsonx::Array)
		{
			TArray< TSharedPtr<FJsonxValue> > ArrayValue = JsonxValue->AsArray();
			int32 ArrLen = ArrayValue.Num();

			// make the output array size match
			FScriptArrayHelper Helper(ArrayProperty, OutValue);
			Helper.Resize(ArrLen);

			// set the property values
			for (int32 i = 0; i < ArrLen; ++i)
			{
				const TSharedPtr<FJsonxValue>& ArrayValueItem = ArrayValue[i];
				if (ArrayValueItem.IsValid() && !ArrayValueItem->IsNull())
				{
					if (!JsonxValueToFPropertyWithContainer(ArrayValueItem, ArrayProperty->Inner, Helper.GetRawPtr(i), ContainerStruct, Container, CheckFlags & (~CPF_ParmFlags), SkipFlags))
					{
						UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable to deserialize array element [%d] for property %s"), i, *Property->GetNameCPP());
						return false;
					}
				}
			}
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import TArray from non-array JSONX key for property %s"), *Property->GetNameCPP());
			return false;
		}
	}
	else if (FMapProperty* MapProperty = CastField<FMapProperty>(Property))
	{
		if (JsonxValue->Type == EJsonx::Object)
		{
			TSharedPtr<FJsonxObject> ObjectValue = JsonxValue->AsObject();

			FScriptMapHelper Helper(MapProperty, OutValue);

			check(ObjectValue);

			int32 MapSize = ObjectValue->Values.Num();
			Helper.EmptyValues(MapSize);

			// set the property values
			for (const auto& Entry : ObjectValue->Values)
			{
				if (Entry.Value.IsValid() && !Entry.Value->IsNull())
				{
					int32 NewIndex = Helper.AddDefaultValue_Invalid_NeedsRehash();

					TSharedPtr<FJsonxValueString> TempKeyValue = MakeShared<FJsonxValueString>(Entry.Key);

					const bool bKeySuccess = JsonxValueToFPropertyWithContainer(TempKeyValue, MapProperty->KeyProp, Helper.GetKeyPtr(NewIndex), ContainerStruct, Container, CheckFlags & (~CPF_ParmFlags), SkipFlags);
					const bool bValueSuccess = JsonxValueToFPropertyWithContainer(Entry.Value, MapProperty->ValueProp, Helper.GetValuePtr(NewIndex), ContainerStruct, Container, CheckFlags & (~CPF_ParmFlags), SkipFlags);

					if (!(bKeySuccess && bValueSuccess))
					{
						UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable to deserialize map element [key: %s] for property %s"), *Entry.Key, *Property->GetNameCPP());
						return false;
					}
				}
			}

			Helper.Rehash();
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import TMap from non-object JSONX key for property %s"), *Property->GetNameCPP());
			return false;
		}
	}
	else if (FSetProperty* SetProperty = CastField<FSetProperty>(Property))
	{
		if (JsonxValue->Type == EJsonx::Array)
		{
			TArray< TSharedPtr<FJsonxValue> > ArrayValue = JsonxValue->AsArray();
			int32 ArrLen = ArrayValue.Num();

			FScriptSetHelper Helper(SetProperty, OutValue);

			// set the property values
			for (int32 i = 0; i < ArrLen; ++i)
			{
				const TSharedPtr<FJsonxValue>& ArrayValueItem = ArrayValue[i];
				if (ArrayValueItem.IsValid() && !ArrayValueItem->IsNull())
				{
					int32 NewIndex = Helper.AddDefaultValue_Invalid_NeedsRehash();
					if (!JsonxValueToFPropertyWithContainer(ArrayValueItem, SetProperty->ElementProp, Helper.GetElementPtr(NewIndex), ContainerStruct, Container, CheckFlags & (~CPF_ParmFlags), SkipFlags))
					{
						UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable to deserialize set element [%d] for property %s"), i, *Property->GetNameCPP());
						return false;
					}
				}
			}

			Helper.Rehash();
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import TSet from non-array JSONX key for property %s"), *Property->GetNameCPP());
			return false;
		}
	}
	else if (FTextProperty* TextProperty = CastField<FTextProperty>(Property))
	{
		if (JsonxValue->Type == EJsonx::String)
		{
			// assume this string is already localized, so import as invariant
			TextProperty->SetPropertyValue(OutValue, FText::FromString(JsonxValue->AsString()));
		}
		else if (JsonxValue->Type == EJsonx::Object)
		{
			TSharedPtr<FJsonxObject> Obj = JsonxValue->AsObject();
			check(Obj.IsValid()); // should not fail if Type == EJsonx::Object

			// import the subvalue as a culture invariant string
			FText Text;
			if (!FJsonxObjectConverter::GetTextFromObject(Obj.ToSharedRef(), Text))
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import FText from JSONX object with invalid keys for property %s"), *Property->GetNameCPP());
				return false;
			}
			TextProperty->SetPropertyValue(OutValue, Text);
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import FText from JSONX that was neither string nor object for property %s"), *Property->GetNameCPP());
			return false;
		}
	}
	else if (FStructProperty *StructProperty = CastField<FStructProperty>(Property))
	{
		static const FName NAME_DateTime(TEXT("DateTime"));
		static const FName NAME_Color(TEXT("Color"));
		static const FName NAME_LinearColor(TEXT("LinearColor"));
		if (JsonxValue->Type == EJsonx::Object)
		{
			TSharedPtr<FJsonxObject> Obj = JsonxValue->AsObject();
			check(Obj.IsValid()); // should not fail if Type == EJsonx::Object
			if (!JsonxAttributesToUStructWithContainer(Obj->Values, StructProperty->Struct, OutValue, ContainerStruct, Container, CheckFlags & (~CPF_ParmFlags), SkipFlags))
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - FJsonxObjectConverter::JsonxObjectToUStruct failed for property %s"), *Property->GetNameCPP());
				return false;
			}
		}
		else if (JsonxValue->Type == EJsonx::String && StructProperty->Struct->GetFName() == NAME_LinearColor)
		{
			FLinearColor& ColorOut = *(FLinearColor*)OutValue;
			FString ColorString = JsonxValue->AsString();

			FColor IntermediateColor;
			IntermediateColor = FColor::FromHex(ColorString);

			ColorOut = IntermediateColor;
		}
		else if (JsonxValue->Type == EJsonx::String && StructProperty->Struct->GetFName() == NAME_Color)
		{
			FColor& ColorOut = *(FColor*)OutValue;
			FString ColorString = JsonxValue->AsString();

			ColorOut = FColor::FromHex(ColorString);
		}
		else if (JsonxValue->Type == EJsonx::String && StructProperty->Struct->GetFName() == NAME_DateTime)
		{
			FString DateString = JsonxValue->AsString();
			FDateTime& DateTimeOut = *(FDateTime*)OutValue;
			if (DateString == TEXT("min"))
			{
				// min representable value for our date struct. Actual date may vary by platform (this is used for sorting)
				DateTimeOut = FDateTime::MinValue();
			}
			else if (DateString == TEXT("max"))
			{
				// max representable value for our date struct. Actual date may vary by platform (this is used for sorting)
				DateTimeOut = FDateTime::MaxValue();
			}
			else if (DateString == TEXT("now"))
			{
				// this value's not really meaningful from json serialization (since we don't know timezone) but handle it anyway since we're handling the other keywords
				DateTimeOut = FDateTime::UtcNow();
			}
			else if (FDateTime::ParseIso8601(*DateString, DateTimeOut))
			{
				// ok
			}
			else if (FDateTime::Parse(DateString, DateTimeOut))
			{
				// ok
			}
			else
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable to import FDateTime for property %s"), *Property->GetNameCPP());
				return false;
			}
		}
		else if (JsonxValue->Type == EJsonx::String && StructProperty->Struct->GetCppStructOps() && StructProperty->Struct->GetCppStructOps()->HasImportTextItem())
		{
			UScriptStruct::ICppStructOps* TheCppStructOps = StructProperty->Struct->GetCppStructOps();

			FString ImportTextString = JsonxValue->AsString();
			const TCHAR* ImportTextPtr = *ImportTextString;
			if (!TheCppStructOps->ImportTextItem(ImportTextPtr, OutValue, PPF_None, nullptr, (FOutputDevice*)GWarn))
			{
				// Fall back to trying the tagged property approach if custom ImportTextItem couldn't get it done
				Property->ImportText(ImportTextPtr, OutValue, PPF_None, nullptr);
			}
		}
		else if (JsonxValue->Type == EJsonx::String)
		{
			FString ImportTextString = JsonxValue->AsString();
			const TCHAR* ImportTextPtr = *ImportTextString;
			Property->ImportText(ImportTextPtr, OutValue, PPF_None, nullptr);
		}
		else
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import UStruct from non-object JSONX key for property %s"), *Property->GetNameCPP());
			return false;
		}
	}
		else if (FObjectProperty *ObjectProperty = CastField<FObjectProperty>(Property))
		{
			if (JsonxValue->Type == EJsonx::Object)
			{
				UObject* Outer = GetTransientPackage();
				if (ContainerStruct->IsChildOf(UObject::StaticClass()))
				{
					Outer = (UObject*)Container;
				}

				TSharedPtr<FJsonxObject> Obj = JsonxValue->AsObject();
				UClass* PropertyClass = ObjectProperty->PropertyClass;

				// If a specific subclass was stored in the Jsonx, use that instead of the PropertyClass
				FString ClassString = Obj->GetStringField(ObjectClassNameKey);
				Obj->RemoveField(ObjectClassNameKey);
				if (!ClassString.IsEmpty())
				{
					UClass* FoundClass = FindObject<UClass>(ANY_PACKAGE, *ClassString);
					if (FoundClass)
					{
						PropertyClass = FoundClass;
					}
				}

				UObject* createdObj = StaticAllocateObject(PropertyClass, Outer, NAME_None, EObjectFlags::RF_NoFlags, EInternalObjectFlags::None, false);
				(*PropertyClass->ClassConstructor)(FObjectInitializer(createdObj, PropertyClass->ClassDefaultObject, false, false));

				ObjectProperty->SetObjectPropertyValue(OutValue, createdObj);

				check(Obj.IsValid()); // should not fail if Type == EJsonx::Object
				if (!JsonxAttributesToUStructWithContainer(Obj->Values, PropertyClass, createdObj, PropertyClass, createdObj, CheckFlags & (~CPF_ParmFlags), SkipFlags))
				{
					UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - FJsonxObjectConverter::JsonxObjectToUStruct failed for property %s"), *Property->GetNameCPP());
					return false;
				}
			}
			else if (JsonxValue->Type == EJsonx::String)
			{
				// Default to expect a string for everything else
				if (Property->ImportText(*JsonxValue->AsString(), OutValue, 0, NULL) == NULL)
				{
					UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable import property type %s from string value for property %s"), *Property->GetClass()->GetName(), *Property->GetNameCPP());
					return false;
				}
			}
		}
	else
	{
		// Default to expect a string for everything else
		if (Property->ImportText(*JsonxValue->AsString(), OutValue, 0, NULL) == NULL)
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Unable import property type %s from string value for property %s"), *Property->GetClass()->GetName(), *Property->GetNameCPP());
			return false;
		}
	}

	return true;
	}


	bool JsonxValueToFPropertyWithContainer(const TSharedPtr<FJsonxValue>& JsonxValue, FProperty* Property, void* OutValue, const UStruct* ContainerStruct, void* Container, int64 CheckFlags, int64 SkipFlags)
	{
		if (!JsonxValue.IsValid())
		{
			UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Invalid value JSONX key"));
			return false;
		}

		bool bArrayOrSetProperty = Property->IsA<FArrayProperty>() || Property->IsA<FSetProperty>();
		bool bJsonxArray = JsonxValue->Type == EJsonx::Array;

		if (!bJsonxArray)
		{
			if (bArrayOrSetProperty)
			{
				UE_LOG(LogJsonx, Error, TEXT("JsonxValueToUProperty - Attempted to import TArray from non-array JSONX key"));
				return false;
			}

			if (Property->ArrayDim != 1)
			{
				UE_LOG(LogJsonx, Warning, TEXT("Ignoring excess properties when deserializing %s"), *Property->GetName());
			}

			return ConvertScalarJsonxValueToFPropertyWithContainer(JsonxValue, Property, OutValue, ContainerStruct, Container, CheckFlags, SkipFlags);
		}

		// In practice, the ArrayDim == 1 check ought to be redundant, since nested arrays of FPropertys are not supported
		if (bArrayOrSetProperty && Property->ArrayDim == 1)
		{
			// Read into TArray
			return ConvertScalarJsonxValueToFPropertyWithContainer(JsonxValue, Property, OutValue, ContainerStruct, Container, CheckFlags, SkipFlags);
		}

		// We're deserializing a JSONX array
		const auto& ArrayValue = JsonxValue->AsArray();
		if (Property->ArrayDim < ArrayValue.Num())
		{
			UE_LOG(LogJsonx, Warning, TEXT("Ignoring excess properties when deserializing %s"), *Property->GetName());
		}

		// Read into native array
		int ItemsToRead = FMath::Clamp(ArrayValue.Num(), 0, Property->ArrayDim);
		for (int Index = 0; Index != ItemsToRead; ++Index)
		{
			if (!ConvertScalarJsonxValueToFPropertyWithContainer(ArrayValue[Index], Property, (char*)OutValue + Index * Property->ElementSize, ContainerStruct, Container, CheckFlags, SkipFlags))
			{
				return false;
			}
		}
		return true;
	}

	bool JsonxAttributesToUStructWithContainer(const TMap< FString, TSharedPtr<FJsonxValue> >& JsonxAttributes, const UStruct* StructDefinition, void* OutStruct, const UStruct* ContainerStruct, void* Container, int64 CheckFlags, int64 SkipFlags)
	{
		if (StructDefinition == FJsonxObjectWrapper::StaticStruct())
		{
			// Just copy it into the object
			FJsonxObjectWrapper* ProxyObject = (FJsonxObjectWrapper*)OutStruct;
			ProxyObject->JsonxObject = MakeShared<FJsonxObject>();
			ProxyObject->JsonxObject->Values = JsonxAttributes;
			return true;
		}

		int32 NumUnclaimedProperties = JsonxAttributes.Num();
		if (NumUnclaimedProperties <= 0)
		{
			return true;
		}

		// iterate over the struct properties
		for (TFieldIterator<FProperty> PropIt(StructDefinition); PropIt; ++PropIt)
		{
			FProperty* Property = *PropIt;

			// Check to see if we should ignore this property
			if (CheckFlags != 0 && !Property->HasAnyPropertyFlags(CheckFlags))
			{
				continue;
			}
			if (Property->HasAnyPropertyFlags(SkipFlags))
			{
				continue;
			}

			// find a json value matching this property name
			const TSharedPtr<FJsonxValue>* JsonxValue = JsonxAttributes.Find(Property->GetName());
			if (!JsonxValue)
			{
				// we allow values to not be found since this mirrors the typical UObject mantra that all the fields are optional when deserializing
				continue;
			}

			if (JsonxValue->IsValid() && !(*JsonxValue)->IsNull())
			{
				void* Value = Property->ContainerPtrToValuePtr<uint8>(OutStruct);
				if (!JsonxValueToFPropertyWithContainer(*JsonxValue, Property, Value, ContainerStruct, Container, CheckFlags, SkipFlags))
				{
					UE_LOG(LogJsonx, Error, TEXT("JsonxObjectToUStruct - Unable to parse %s.%s from JSONX"), *StructDefinition->GetName(), *Property->GetName());
					return false;
				}
			}

			if (--NumUnclaimedProperties <= 0)
			{
				// If we found all properties that were in the JsonxAttributes map, there is no reason to keep looking for more.
				break;
			}
		}

		return true;
	}
}

bool FJsonxObjectConverter::JsonxValueToUProperty(const TSharedPtr<FJsonxValue>& JsonxValue, FProperty* Property, void* OutValue, int64 CheckFlags, int64 SkipFlags)
{
	return JsonxValueToFPropertyWithContainer(JsonxValue, Property, OutValue, nullptr, nullptr, CheckFlags, SkipFlags);
}

bool FJsonxObjectConverter::JsonxObjectToUStruct(const TSharedRef<FJsonxObject>& JsonxObject, const UStruct* StructDefinition, void* OutStruct, int64 CheckFlags, int64 SkipFlags)
{
	return JsonxAttributesToUStruct(JsonxObject->Values, StructDefinition, OutStruct, CheckFlags, SkipFlags);
}

bool FJsonxObjectConverter::JsonxAttributesToUStruct(const TMap< FString, TSharedPtr<FJsonxValue> >& JsonxAttributes, const UStruct* StructDefinition, void* OutStruct, int64 CheckFlags, int64 SkipFlags)
{
	return JsonxAttributesToUStructWithContainer(JsonxAttributes, StructDefinition, OutStruct, StructDefinition, OutStruct, CheckFlags, SkipFlags);
}

//static 
bool FJsonxObjectConverter::GetTextFromField(const FString& FieldName, const TSharedPtr<FJsonxValue>& FieldValue, FText& TextOut)
{
	if (FieldValue.IsValid())
	{
		switch (FieldValue->Type)
		{
			case EJsonx::Number:
			{
				// number
				TextOut = FText::AsNumber(FieldValue->AsNumber());
				return true;
			}
			case EJsonx::String:
			{
				if (FieldName.StartsWith(TEXT("date-")))
				{
					FDateTime Dte;
					if (FDateTime::ParseIso8601(*FieldValue->AsString(), Dte))
					{
						TextOut = FText::AsDate(Dte);
						return true;
					}
				}
				else if (FieldName.StartsWith(TEXT("datetime-")))
				{
					FDateTime Dte;
					if (FDateTime::ParseIso8601(*FieldValue->AsString(), Dte))
					{
						TextOut = FText::AsDateTime(Dte);
						return true;
					}
				}
				else
				{
				// culture invariant string
					TextOut = FText::FromString(FieldValue->AsString());
					return true;
				}
				break;
			}
			case EJsonx::Object:
			{
				// localized string
				if (FJsonxObjectConverter::GetTextFromObject(FieldValue->AsObject().ToSharedRef(), TextOut))
				{
					return true;
				}

				UE_LOG(LogJsonx, Error, TEXT("Unable to apply Jsonx parameter %s (could not parse object)"), *FieldName);
				break;
			}
			default:
			{
				UE_LOG(LogJsonx, Error, TEXT("Unable to apply Jsonx parameter %s (bad type)"), *FieldName);
				break;
			}
		}
	}
	return false;
}

FFormatNamedArguments FJsonxObjectConverter::ParseTextArgumentsFromJsonx(const TSharedPtr<const FJsonxObject>& JsonxObject)
{
	FFormatNamedArguments NamedArgs;
	if (JsonxObject.IsValid())
	{
		for (const auto& It : JsonxObject->Values)
		{
			FText TextValue;
			if (GetTextFromField(It.Key, It.Value, TextValue))
			{
				NamedArgs.Emplace(It.Key, TextValue);
			}
		}
	}
	return NamedArgs;
}