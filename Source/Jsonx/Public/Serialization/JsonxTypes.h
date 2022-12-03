// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"

class Error;

/**
 * Jsonx (JavaScript Object Notation) is a lightweight data-interchange format.
 * Information on how it works can be found here: http://www.json.org/.
 * This code was written from scratch with only the Jsonx spec as a guide.
 *
 * In order to use Jsonx effectively, you need to be familiar with the Object/Value
 * hierarchy, and you should use the FJsonxObject class and FJsonxValue subclasses.
 */

/**
 * Represents all the types a Jsonx Value can be.
 */
enum class EJsonx
{
	None,
	Null,
	String,
	Number,
	Boolean,
	Array,
	Object
};


enum class EJsonxToken
{
	None,
	Comma,
	CurlyOpen,
	CurlyClose,
	SquareOpen,
	SquareClose,
	Colon,
	String,

	// short values
	Number,
	True,
	False,
	Null,

	Identifier
};

FORCEINLINE bool EJsonxToken_IsShortValue(EJsonxToken Token)
{
	return Token >= EJsonxToken::Number && Token <= EJsonxToken::Null;
}

enum class EJsonxNotation
{
	ObjectStart,
	ObjectEnd,
	ArrayStart,
	ArrayEnd,
	Boolean,
	String,
	Number,
	Null,
	Error
};
