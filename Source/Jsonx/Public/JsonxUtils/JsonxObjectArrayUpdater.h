// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonxObject.h"
#include "Dom/JsonxValue.h"

/**
 * Utility to update an array of json objects from an array of elements (of arbitrary type).
 * Elements in the source array and the destination json object array are matched based on an
 * arbitrary key (provided by the FGetElementKey and FTryGetJsonxObjectKey delegates respectively).
 * Existing elements get "updated" via the FUpdateJsonxObject delegate. The update scheme is entirely 
 * customizable; for example, it can be non-destructive and leave some json fields unchanged.
 * Elements from the source array that are not in the json array (based on the "key") are added to it.
 * Elements that are not present in the source array (based on the "key") are removed from the json array.
 * If the source array is empty the json array field is removed.
 */
template <typename ElementType, typename KeyType>
struct FJsonxObjectArrayUpdater
{
	DECLARE_DELEGATE_RetVal_OneParam(KeyType, FGetElementKey, const ElementType&);

	DECLARE_DELEGATE_RetVal_TwoParams(bool, FTryGetJsonxObjectKey, const FJsonxObject&, KeyType& /*OutKey*/);

	DECLARE_DELEGATE_TwoParams(FUpdateJsonxObject, const ElementType&, FJsonxObject&);

	static void Execute(FJsonxObject& JsonxObject, const FString& ArrayName, const TArray<ElementType>& SourceArray, FGetElementKey GetElementKey, FTryGetJsonxObjectKey TryGetJsonxObjectKey, FUpdateJsonxObject UpdateJsonxObject)
	{
		if (SourceArray.Num() > 0)
		{
			TArray<TSharedPtr<FJsonxValue>> NewJsonxValues;
			{
				const TArray<TSharedPtr<FJsonxValue>>* ExistingJsonxValues;
				if (JsonxObject.TryGetArrayField(ArrayName, ExistingJsonxValues))
				{
					// Build a map of elements for quick access and to keep track of which ones got updated
					TMap<KeyType, const ElementType*> ElementsMap;
					for (const ElementType& Element : SourceArray)
					{
						ElementsMap.Add(GetElementKey.Execute(Element), &Element);
					}

					// Update existing json values and discard entries that no longer exist or are invalid
					for (TSharedPtr<FJsonxValue> ExistingJsonxValue : *ExistingJsonxValues)
					{
						const TSharedPtr<FJsonxObject>* ExistingJsonxValueAsObject;
						if (ExistingJsonxValue->TryGetObject(ExistingJsonxValueAsObject))
						{
							KeyType ElementKey;
							if (TryGetJsonxObjectKey.Execute(**ExistingJsonxValueAsObject, ElementKey))
							{
								if (const ElementType** ElementPtr = ElementsMap.Find(ElementKey))
								{
									UpdateJsonxObject.Execute(**ElementPtr, **ExistingJsonxValueAsObject);
									NewJsonxValues.Add(ExistingJsonxValue);
									ElementsMap.Remove(ElementKey);
								}
							}
						}
					}

					// Add new elements
					for (auto It = ElementsMap.CreateConstIterator(); It; ++It)
					{
						TSharedPtr<FJsonxObject> NewJsonxObject = MakeShareable(new FJsonxObject);
						UpdateJsonxObject.Execute(*It.Value(), *NewJsonxObject.Get());
						NewJsonxValues.Add(MakeShareable(new FJsonxValueObject(NewJsonxObject)));
					}
				}
				else
				{
					// Array doesn't exist in the given JsonxObject, so build a new array
					for (const ElementType& Element : SourceArray)
					{
						TSharedPtr<FJsonxObject> NewJsonxObject = MakeShareable(new FJsonxObject);
						UpdateJsonxObject.Execute(Element, *NewJsonxObject.Get());
						NewJsonxValues.Add(MakeShareable(new FJsonxValueObject(NewJsonxObject)));
					}
				}
			}

			// Set the new content of the json array
			JsonxObject.SetArrayField(ArrayName, NewJsonxValues);
		}
		else
		{
			// Source array is empty so remove the json array
			JsonxObject.RemoveField(ArrayName);
		}
	}
};