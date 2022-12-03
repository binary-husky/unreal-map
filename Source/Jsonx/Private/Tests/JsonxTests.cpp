// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"
#include "Policies/CondensedJsonxPrintPolicy.h"
#include "Serialization/JsonxTypes.h"
#include "Serialization/JsonxReader.h"
#include "Policies/PrettyJsonxPrintPolicy.h"
#include "Serialization/JsonxSerializer.h"

#if WITH_DEV_AUTOMATION_TESTS

/**
 * FJsonxAutomationTest
 * Simple unit test that runs Jsonx's in-built test cases
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FJsonxAutomationTest, "System.Engine.FileSystem.JSONX", EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter )

typedef TJsonxWriterFactory< TCHAR, TCondensedJsonxPrintPolicy<TCHAR> > FCondensedJsonxStringWriterFactory;
typedef TJsonxWriter< TCHAR, TCondensedJsonxPrintPolicy<TCHAR> > FCondensedJsonxStringWriter;

typedef TJsonxWriterFactory< TCHAR, TPrettyJsonxPrintPolicy<TCHAR> > FPrettyJsonxStringWriterFactory;
typedef TJsonxWriter< TCHAR, TPrettyJsonxPrintPolicy<TCHAR> > FPrettyJsonxStringWriter;

/** 
 * Execute the Jsonx test cases
 *
 * @return	true if the test was successful, false otherwise
 */
bool FJsonxAutomationTest::RunTest(const FString& Parameters)
{
	// Null Case
	{
		const FString InputString = TEXT("");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		check( FJsonxSerializer::Deserialize( Reader, Object ) == false );
		check( !Object.IsValid() );
	}

	// Empty Object Case
	{
		const FString InputString = TEXT("{}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		check( FJsonxSerializer::Deserialize( Reader, Object ) );
		check( Object.IsValid() );

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );
		check( InputString == OutputString );
	}

	// Empty Array Case
	{
		const FString InputString = TEXT("[]");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TArray< TSharedPtr<FJsonxValue> > Array;
		check( FJsonxSerializer::Deserialize( Reader, Array ) );
		check( Array.Num() == 0 );

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Array, Writer ) );
		check( InputString == OutputString );
	}

	// Simple Array Case
	{
		const FString InputString = 
			TEXT("[")
			TEXT(	"{")
			TEXT(		"\"Value\":\"Some String\"")
			TEXT(	"}")
			TEXT("]");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TArray< TSharedPtr<FJsonxValue> > Array;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Array);
		check(bSuccessful);
		check( Array.Num() == 1 );
		check( Array[0].IsValid() );

		TSharedPtr< FJsonxObject > Object = Array[0]->AsObject();
		check( Object.IsValid() );
		check( Object->GetStringField( TEXT("Value") ) == TEXT("Some String") );

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Array, Writer ) );
		check( InputString == OutputString );
	}

	// Object Array Case
	{
		const FString InputString =
			TEXT("[")
			TEXT(	"{")
			TEXT(		"\"Value\":\"Some String1\"")
			TEXT(	"},")
			TEXT(	"{")
			TEXT(		"\"Value\":\"Some String2\"")
			TEXT(	"},")
			TEXT(	"{")
			TEXT(		"\"Value\":\"Some String3\"")
			TEXT(	"}")
			TEXT("]");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create(InputString);

		TArray< TSharedPtr<FJsonxValue> > Array;

		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Array);
		check(bSuccessful);
		check(Array.Num() == 3);
		check(Array[0].IsValid());
		check(Array[1].IsValid());
		check(Array[2].IsValid());

		TSharedPtr< FJsonxObject > Object = Array[0]->AsObject();
		check(Object.IsValid());
		check(Object->GetStringField(TEXT("Value")) == TEXT("Some String1"));

		Object = Array[1]->AsObject();
		check(Object.IsValid());
		check(Object->GetStringField(TEXT("Value")) == TEXT("Some String2"));

		Object = Array[2]->AsObject();
		check(Object.IsValid());
		check(Object->GetStringField(TEXT("Value")) == TEXT("Some String3"));

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create(&OutputString);
		check(FJsonxSerializer::Serialize(Array, Writer));
		check(InputString == OutputString);
	}

	// Number Array Case
	{
		const FString InputString =
			TEXT("[")
			TEXT("10,")
			TEXT("20,")
			TEXT("30,")
			TEXT("40")
			TEXT("]");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create(InputString);

		TArray< TSharedPtr<FJsonxValue> > Array;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Array);
		check(bSuccessful);
		check(Array.Num() == 4);
		check(Array[0].IsValid());
		check(Array[1].IsValid());
		check(Array[2].IsValid());
		check(Array[3].IsValid());

		double Number = Array[0]->AsNumber();
		check(Number == 10);

		Number = Array[1]->AsNumber();
		check(Number == 20);

		Number = Array[2]->AsNumber();
		check(Number == 30);

		Number = Array[3]->AsNumber();
		check(Number == 40);

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create(&OutputString);
		check(FJsonxSerializer::Serialize(Array, Writer));
		check(InputString == OutputString);
	}

	// String Array Case
	{
		const FString InputString =
			TEXT("[")
			TEXT("\"Some String1\",")
			TEXT("\"Some String2\",")
			TEXT("\"Some String3\",")
			TEXT("\"Some String4\"")
			TEXT("]");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create(InputString);

		TArray< TSharedPtr<FJsonxValue> > Array;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Array);
		check(bSuccessful);
		check(Array.Num() == 4);
		check(Array[0].IsValid());
		check(Array[1].IsValid());
		check(Array[2].IsValid());
		check(Array[3].IsValid());

		FString Text = Array[0]->AsString();
		check(Text == TEXT("Some String1"));

		Text = Array[1]->AsString();
		check(Text == TEXT("Some String2"));

		Text = Array[2]->AsString();
		check(Text == TEXT("Some String3"));

		Text = Array[3]->AsString();
		check(Text == TEXT("Some String4"));

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create(&OutputString);
		check(FJsonxSerializer::Serialize(Array, Writer));
		check(InputString == OutputString);
	}

	// Complex Array Case
	{
		const FString InputString =
			TEXT("[")
			TEXT(	"\"Some String1\",")
			TEXT(	"10,")
			TEXT(	"{")
			TEXT(		"\"Value\":\"Some String3\"")
			TEXT(	"},")
			TEXT(	"[")
			TEXT(		"\"Some String4\",")
			TEXT(		"\"Some String5\"")
			TEXT(	"],")
			TEXT(	"true,")
			TEXT(	"null")
			TEXT("]");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create(InputString);

		TArray< TSharedPtr<FJsonxValue> > Array;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Array);
		check(bSuccessful);
		check(Array.Num() == 6);
		check(Array[0].IsValid());
		check(Array[1].IsValid());
		check(Array[2].IsValid());
		check(Array[3].IsValid());
		check(Array[4].IsValid());
		check(Array[5].IsValid());

		FString Text = Array[0]->AsString();
		check(Text == TEXT("Some String1"));

		double Number = Array[1]->AsNumber();
		check(Number == 10);

		TSharedPtr< FJsonxObject > Object = Array[2]->AsObject();
		check(Object.IsValid());
		check(Object->GetStringField(TEXT("Value")) == TEXT("Some String3"));

		const TArray<TSharedPtr< FJsonxValue >>& InnerArray = Array[3]->AsArray();
		check(InnerArray.Num() == 2);
		check(Array[0].IsValid());
		check(Array[1].IsValid());

		Text = InnerArray[0]->AsString();
		check(Text == TEXT("Some String4"));

		Text = InnerArray[1]->AsString();
		check(Text == TEXT("Some String5"));

		bool Boolean = Array[4]->AsBool();
		check(Boolean == true);

		check(Array[5]->IsNull() == true);

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create(&OutputString);
		check(FJsonxSerializer::Serialize(Array, Writer));
		check(InputString == OutputString);
	}

	// String Test
	{
		const FString InputString =
			TEXT("{")
			TEXT(	"\"Value\":\"Some String, Escape Chars: \\\\, \\\", \\/, \\b, \\f, \\n, \\r, \\t, \\u002B\"")
			TEXT("}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Object);
		check(bSuccessful);
		check( Object.IsValid() );

		const TSharedPtr<FJsonxValue>* Value = Object->Values.Find(TEXT("Value"));
		check(Value && (*Value)->Type == EJsonx::String);
		const FString String = (*Value)->AsString();
		check(String == TEXT("Some String, Escape Chars: \\, \", /, \b, \f, \n, \r, \t, +"));

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );

		const FString TestOutput =
			TEXT("{")
			TEXT(	"\"Value\":\"Some String, Escape Chars: \\\\, \\\", /, \\b, \\f, \\n, \\r, \\t, +\"")
			TEXT("}");
		check(OutputString == TestOutput);
	}

	//// Number Test
	//{
	//	const FString InputString =
	//		TEXT("{")
	//		TEXT(	"\"Value1\":2.544e+15,")
	//		TEXT(	"\"Value2\":-0.544E-2,")
	//		TEXT(	"\"Value3\":251e3,")
	//		TEXT(	"\"Value4\":-0.0,")
	//		TEXT(	"\"Value5\":843")
	//		TEXT("}");
	//	TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

	//	TSharedPtr<FJsonxObject> Object;
	//	bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Object);
	//	check(bSuccessful);
	//	check( Object.IsValid() );

	//	double TestValues[] = {2.544e+15, -0.544e-2, 251e3, -0.0, 843};
	//	for (int32 i = 0; i < 5; ++i)
	//	{
	//		const TSharedPtr<FJsonxValue>* Value = Object->Values.Find(FString::Printf(TEXT("Value%i"), i + 1));
	//		check(Value && (*Value)->Type == EJsonx::Number);
	//		const double Number = (*Value)->AsNumber();
	//		check(Number == TestValues[i]);
	//	}

	//	FString OutputString;
	//	TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
	//	check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );

	//	// %g isn't standardized, so we use the same %g format that is used inside PrintJsonx instead of hardcoding the values here
	//	const FString TestOutput = FString::Printf(
	//		TEXT("{")
	//		TEXT(	"\"Value1\":%.17g,")
	//		TEXT(	"\"Value2\":%.17g,")
	//		TEXT(	"\"Value3\":%.17g,")
	//		TEXT(	"\"Value4\":%.17g,")
	//		TEXT(	"\"Value5\":%.17g")
	//		TEXT("}"),
	//		TestValues[0], TestValues[1], TestValues[2], TestValues[3], TestValues[4]);
	//	check(OutputString == TestOutput);
	//}

	// Boolean/Null Test
	{
		const FString InputString =
			TEXT("{")
			TEXT(	"\"Value1\":true,")
			TEXT(	"\"Value2\":true,")
			TEXT(	"\"Value3\":faLsE,")
			TEXT(	"\"Value4\":null,")
			TEXT(	"\"Value5\":NULL")
			TEXT("}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Object);
		check(bSuccessful);
		check( Object.IsValid() );

		bool TestValues[] = {true, true, false};
		for (int32 i = 0; i < 5; ++i)
		{
			const TSharedPtr<FJsonxValue>* Value = Object->Values.Find(FString::Printf(TEXT("Value%i"), i + 1));
			check(Value);
			if (i < 3)
			{
				check((*Value)->Type == EJsonx::Boolean);
				const bool Bool = (*Value)->AsBool();
				check(Bool == TestValues[i]);
			}
			else
			{
				check((*Value)->Type == EJsonx::Null);
				check((*Value)->IsNull());
			}
		}

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );

		const FString TestOutput =
			TEXT("{")
			TEXT(	"\"Value1\":true,")
			TEXT(	"\"Value2\":true,")
			TEXT(	"\"Value3\":false,")
			TEXT(	"\"Value4\":null,")
			TEXT(	"\"Value5\":null")
			TEXT("}");
		check(OutputString == TestOutput);
	}

	// Object Test && extra whitespace test
	{
		const FString InputStringWithExtraWhitespace =
			TEXT("		\n\r\n	   {")
			TEXT(	"\"Object\":")
			TEXT(	"{")
			TEXT(		"\"NestedValue\":null,")
			TEXT(		"\"NestedObject\":{}")
			TEXT(	"},")
			TEXT(	"\"Value\":true")
			TEXT("}		\n\r\n	   ");

		const FString InputString =
			TEXT("{")
			TEXT(	"\"Object\":")
			TEXT(	"{")
			TEXT(		"\"NestedValue\":null,")
			TEXT(		"\"NestedObject\":{}")
			TEXT(	"},")
			TEXT(	"\"Value\":true")
			TEXT("}");

		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputStringWithExtraWhitespace );

		TSharedPtr<FJsonxObject> Object;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Object);
		check(bSuccessful);
		check( Object.IsValid() );

		const TSharedPtr<FJsonxValue>* InnerValueFail = Object->Values.Find(TEXT("InnerValue"));
		check(!InnerValueFail);

		const TSharedPtr<FJsonxValue>* ObjectValue = Object->Values.Find(TEXT("Object"));
		check(ObjectValue && (*ObjectValue)->Type == EJsonx::Object);
		const TSharedPtr<FJsonxObject> InnerObject = (*ObjectValue)->AsObject();
		check(InnerObject.IsValid());

		{
			const TSharedPtr<FJsonxValue>* NestedValueValue = InnerObject->Values.Find(TEXT("NestedValue"));
			check(NestedValueValue && (*NestedValueValue)->Type == EJsonx::Null);
			check((*NestedValueValue)->IsNull());

			const TSharedPtr<FJsonxValue>* NestedObjectValue = InnerObject->Values.Find(TEXT("NestedObject"));
			check(NestedObjectValue && (*NestedObjectValue)->Type == EJsonx::Object);
			const TSharedPtr<FJsonxObject> InnerInnerObject = (*NestedObjectValue)->AsObject();
			check(InnerInnerObject.IsValid());

			{
				const TSharedPtr<FJsonxValue>* NestedValueValueFail = InnerInnerObject->Values.Find(TEXT("NestedValue"));
				check(!NestedValueValueFail);
			}
		}

		const TSharedPtr<FJsonxValue>* ValueValue = Object->Values.Find(TEXT("Value"));
		check(ValueValue && (*ValueValue)->Type == EJsonx::Boolean);
		const bool Bool = (*ValueValue)->AsBool();
		check(Bool);

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );
		check(OutputString == InputString);
	}

	// Array Test
	{
		const FString InputString =
			TEXT("{")
			TEXT(	"\"Array\":")
			TEXT(	"[")
			TEXT(		"[],")
			TEXT(		"\"Some String\",")
			TEXT(		"\"Another String\",")
			TEXT(		"null,")
			TEXT(		"true,")
			TEXT(		"false,")
			TEXT(		"45,")
			TEXT(		"{}")
			TEXT(	"]")
			TEXT("}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		bool bSuccessful = FJsonxSerializer::Deserialize(Reader, Object);
		check(bSuccessful);
		check( Object.IsValid() );

		const TSharedPtr<FJsonxValue>* InnerValueFail = Object->Values.Find(TEXT("InnerValue"));
		check(!InnerValueFail);

		const TSharedPtr<FJsonxValue>* ArrayValue = Object->Values.Find(TEXT("Array"));
		check(ArrayValue && (*ArrayValue)->Type == EJsonx::Array);
		const TArray< TSharedPtr<FJsonxValue> > Array = (*ArrayValue)->AsArray();
		check(Array.Num() == 8);

		EJsonx ValueTypes[] = {EJsonx::Array, EJsonx::String, EJsonx::String, EJsonx::Null,
			EJsonx::Boolean, EJsonx::Boolean, EJsonx::Number, EJsonx::Object};
		for (int32 i = 0; i < Array.Num(); ++i)
		{
			const TSharedPtr<FJsonxValue>& Value = Array[i];
			check(Value.IsValid());
			check(Value->Type == ValueTypes[i]);
		}

		const TArray< TSharedPtr<FJsonxValue> >& InnerArray = Array[0]->AsArray();
		check(InnerArray.Num() == 0);
		check(Array[1]->AsString() == TEXT("Some String"));
		check(Array[2]->AsString() == TEXT("Another String"));
		check(Array[3]->IsNull());
		check(Array[4]->AsBool());
		check(!Array[5]->AsBool());
		check(FMath::Abs(Array[6]->AsNumber() - 45.f) < KINDA_SMALL_NUMBER);
		const TSharedPtr<FJsonxObject> InnerObject = Array[7]->AsObject();
		check(InnerObject.IsValid());

		FString OutputString;
		TSharedRef< FCondensedJsonxStringWriter > Writer = FCondensedJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );
		check(OutputString == InputString);
	}

	// Pretty Print Test
	{
		const FString InputString =
			TEXT("{")									LINE_TERMINATOR
			TEXT("	\"Data1\": \"value\",")				LINE_TERMINATOR
			TEXT("	\"Data2\": \"value\",")				LINE_TERMINATOR
			TEXT("	\"Array\": [")						LINE_TERMINATOR
			TEXT("		{")								LINE_TERMINATOR
			TEXT("			\"InnerData1\": \"value\"")	LINE_TERMINATOR
			TEXT("		},")							LINE_TERMINATOR
			TEXT("		[],")							LINE_TERMINATOR
			TEXT("		[ 1, 2, 3, 4 ],")				LINE_TERMINATOR
			TEXT("		{")								LINE_TERMINATOR
			TEXT("		},")							LINE_TERMINATOR
			TEXT("		\"value\",")					LINE_TERMINATOR
			TEXT("		\"value\"")						LINE_TERMINATOR
			TEXT("	],")								LINE_TERMINATOR
			TEXT("	\"Object\":")						LINE_TERMINATOR
			TEXT("	{")									LINE_TERMINATOR
			TEXT("	}")									LINE_TERMINATOR
			TEXT("}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		TSharedPtr<FJsonxObject> Object;
		check( FJsonxSerializer::Deserialize( Reader, Object ) );
		check( Object.IsValid() );

		FString OutputString;
		TSharedRef< FPrettyJsonxStringWriter > Writer = FPrettyJsonxStringWriterFactory::Create( &OutputString );
		check( FJsonxSerializer::Serialize( Object.ToSharedRef(), Writer ) );
		check(OutputString == InputString);
	}
	  
	// Line and Character # test
	{
		const FString InputString =
			TEXT("{")									LINE_TERMINATOR
			TEXT("	\"Data1\": \"value\",")				LINE_TERMINATOR
			TEXT("	\"Array\":")						LINE_TERMINATOR
			TEXT("	[")									LINE_TERMINATOR
			TEXT("		12345,")						LINE_TERMINATOR
			TEXT("		True")							LINE_TERMINATOR
			TEXT("	],")								LINE_TERMINATOR
			TEXT("	\"Object\":")						LINE_TERMINATOR
			TEXT("	{")									LINE_TERMINATOR
			TEXT("	}")									LINE_TERMINATOR
			TEXT("}");
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( InputString );

		EJsonxNotation Notation = EJsonxNotation::Null;
		check( Reader->ReadNext( Notation ) && Notation == EJsonxNotation::ObjectStart );
		check( Reader->GetLineNumber() == 1 && Reader->GetCharacterNumber() == 1 );

		check( Reader->ReadNext( Notation ) && Notation == EJsonxNotation::String );
		check( Reader->GetLineNumber() == 2 && Reader->GetCharacterNumber() == 17 );

		check( Reader->ReadNext( Notation ) && Notation == EJsonxNotation::ArrayStart );
		check( Reader->GetLineNumber() == 4 && Reader->GetCharacterNumber() == 2 );

		check( Reader->ReadNext( Notation ) && Notation == EJsonxNotation::Number );
		check( Reader->GetLineNumber() == 5 && Reader->GetCharacterNumber() == 7 );

		check( Reader->ReadNext( Notation ) && Notation == EJsonxNotation::Boolean );
		check( Reader->GetLineNumber() == 6 && Reader->GetCharacterNumber() == 6 );
	}

	// Failure Cases
	TArray<FString> FailureInputs;

	// Unclosed Object
	FailureInputs.Add(
		TEXT("{"));

	// Values in Object without identifiers
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Value1\",")
		TEXT(	"\"Value2\",")
		TEXT(	"43")
		TEXT("}"));

	// Unexpected End Of Input Found
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Object\":")
		TEXT(	"{")
		TEXT(		"\"NestedValue\":null,"));

	// Missing first brace
	FailureInputs.Add(
		TEXT(	"\"Object\":")
		TEXT(		"{")
		TEXT(		"\"NestedValue\":null,")
		TEXT(		"\"NestedObject\":{}")
		TEXT(	"},")
		TEXT(	"\"Value\":true")
		TEXT("}"));

	// Missing last character
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Object\":")
		TEXT(	"{")
		TEXT(		"\"NestedValue\":null,")
		TEXT(		"\"NestedObject\":{}")
		TEXT(	"},")
		TEXT(	"\"Value\":true"));

	// Missing curly brace
	FailureInputs.Add(TEXT("}"));

	// Missing bracket
	FailureInputs.Add(TEXT("]"));

	// Extra last character
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Object\":")
		TEXT(	"{")
		TEXT(		"\"NestedValue\":null,")
		TEXT(		"\"NestedObject\":{}")
		TEXT(	"},")
		TEXT(	"\"Value\":true")
		TEXT("}0"));

	// Missing comma
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Value1\":null,")
		TEXT(	"\"Value2\":\"string\"")
		TEXT(	"\"Value3\":65.3")
		TEXT("}"));

	// Extra comma
	FailureInputs.Add(
		TEXT("{")
		TEXT(	"\"Value1\":null,")
		TEXT(	"\"Value2\":\"string\",")
		TEXT(	"\"Value3\":65.3,")
		TEXT("}"));

	// Badly formed true/false/null
	FailureInputs.Add(TEXT("{\"Value\":tru}"));
	FailureInputs.Add(TEXT("{\"Value\":full}"));
	FailureInputs.Add(TEXT("{\"Value\":nulle}"));
	FailureInputs.Add(TEXT("{\"Value\":n%ll}"));

	// Floating Point Failures
	FailureInputs.Add(TEXT("{\"Value\":65.3e}"));
	FailureInputs.Add(TEXT("{\"Value\":65.}"));
	FailureInputs.Add(TEXT("{\"Value\":.7}"));
	FailureInputs.Add(TEXT("{\"Value\":+6}"));
	FailureInputs.Add(TEXT("{\"Value\":01}"));
	FailureInputs.Add(TEXT("{\"Value\":00.56}"));
	FailureInputs.Add(TEXT("{\"Value\":-1.e+4}"));
	FailureInputs.Add(TEXT("{\"Value\":2e+}"));

	// Bad Escape Characters
	FailureInputs.Add(TEXT("{\"Value\":\"Hello\\xThere\"}"));
	FailureInputs.Add(TEXT("{\"Value\":\"Hello\\u123There\"}"));
	FailureInputs.Add(TEXT("{\"Value\":\"Hello\\RThere\"}"));

	for (int32 i = 0; i < FailureInputs.Num(); ++i)
	{
		TSharedRef< TJsonxReader<> > Reader = TJsonxReaderFactory<>::Create( FailureInputs[i] );

		TSharedPtr<FJsonxObject> Object;
		check( FJsonxSerializer::Deserialize( Reader, Object ) == false );
		check( !Object.IsValid() );
	}

	// TryGetNumber tests
	{
		auto JsonxNumberToInt64 = [](double Val, int64& OutVal) -> bool
		{
			FJsonxValueNumber JsonxVal(Val);
			return ((FJsonxValue&)JsonxVal).TryGetNumber(OutVal);
		};

		auto JsonxNumberToInt32 = [](double Val, int32& OutVal) -> bool
		{
			FJsonxValueNumber JsonxVal(Val);
			return ((FJsonxValue&)JsonxVal).TryGetNumber(OutVal);
		};

		auto JsonxNumberToUInt32 = [](double Val, uint32& OutVal) -> bool
		{
			FJsonxValueNumber JsonxVal(Val);
			return ((FJsonxValue&)JsonxVal).TryGetNumber(OutVal);
		};
		
		// TryGetNumber-Int64 tests
		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(9007199254740991.0, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Big Float64 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Big Float64"), IntVal, 9007199254740991LL);
		}

		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(-9007199254740991.0, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Small Float64 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Small Float64"), IntVal, -9007199254740991LL);
		}

		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(0.4999999999999997, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Lesser than near half succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Lesser than near half rounds to zero"), IntVal, 0LL);
		}

		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(-0.4999999999999997, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Greater than near negative half succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Greater than near negative half rounds to zero"), IntVal, 0LL);
		}

		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(0.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Half rounds to next integer succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Half rounds to next integer"), IntVal, 1LL);
		}

		{
			int64 IntVal;
			bool bOk = JsonxNumberToInt64(-0.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int64 Negative half rounds to next negative integer succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int64 Negative half rounds to next negative integer succeeds"), IntVal, -1LL);
		}

		// TryGetNumber-Int32 tests
		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(2147483647.000001, IntVal);
			TestFalse(TEXT("TryGetNumber-Int32 Number greater than max Int32 fails"), bOk);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(-2147483648.000001, IntVal);
			TestFalse(TEXT("TryGetNumber-Int32 Number lesser than min Int32 fails"), bOk);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(2147483647.0, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Max Int32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Max Int32"), IntVal, INT_MAX);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(2147483646.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Round up to max Int32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Round up to max Int32"), IntVal, INT_MAX);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(-2147483648.0, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Min Int32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Min Int32"), IntVal, INT_MIN);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(-2147483647.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Round down to min Int32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Round down to min Int32"), IntVal, INT_MIN);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(0.4999999999999997, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Lesser than near half succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Lesser than near half rounds to zero"), IntVal, 0);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(-0.4999999999999997, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Greater than near negative half succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Greater than near negative half rounds to zero"), IntVal, 0);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(0.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Half rounds to next integer succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Half rounds to next integer"), IntVal, 1);
		}

		{
			int32 IntVal;
			bool bOk = JsonxNumberToInt32(-0.5, IntVal);
			TestTrue(TEXT("TryGetNumber-Int32 Negative half rounds to next negative integer succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-Int32 Negative half rounds to next negative integer succeeds"), IntVal, -1);
		}

		// TryGetNumber-UInt32 tests
		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(4294967295.000001, IntVal);
			TestFalse(TEXT("TryGetNumber-UInt32 Number greater than max Uint32 fails"), bOk);
		}

		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(-0.000000000000001, IntVal);
			TestFalse(TEXT("TryGetNumber-UInt32 Negative number fails"), bOk);
		}

		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(4294967295.0, IntVal);
			TestTrue(TEXT("TryGetNumber-UInt32 Max UInt32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-UInt32  Max UInt32"), IntVal, UINT_MAX);
		}

		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(4294967294.5, IntVal);
			TestTrue(TEXT("TryGetNumber-UInt32 Round up to max UInt32 succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-UInt32 Round up to max UInt32"), IntVal, UINT_MAX);
		}

		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(0.4999999999999997, IntVal);
			TestTrue(TEXT("TryGetNumber-UInt32 Lesser than near half succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-UInt32 Lesser than near half rounds to zero"), IntVal, 0U);
		}

		{
			uint32 IntVal;
			bool bOk = JsonxNumberToUInt32(0.5, IntVal);
			TestTrue(TEXT("TryGetNumber-UInt32 Half rounds to next integer succeeds"), bOk);
			TestEqual(TEXT("TryGetNumber-UInt32 Half rounds to next integer"), IntVal, 1U);
		}
	}

	return true;
}

#endif //WITH_DEV_AUTOMATION_TESTS
