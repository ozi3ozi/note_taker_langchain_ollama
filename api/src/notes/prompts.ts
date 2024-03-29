import { ChatPromptTemplate, PromptTemplate } from "langchain/prompts";
import { BaseMessageChunk } from "langchain/schema";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

//For OllamaFunctions
export const toolSystemPromptTemplate = `You have access to the following tools:

{tools}

To use a tool, respond with a JSON object with the following structure:
{{
  "tool": <name of the called tool>,
  "tool_input": <parameters for the tool matching the above JSON schema>
}}`;

const notesPropsSchema = z.object({
    notes: z.array(
        z.object({
        note: z.string().describe("The notes"),
        pageNumbers: z.array(z.number().describe("The page number(s) of the note")),
    })
    ),
});

export const NOTES_TOOL_SCHEMA = {
    name: "formatNotes",
    description: "Format the notes response",
    parameters: {
        type: "object",
        properties: {
            notes: zodToJsonSchema(notesPropsSchema),
        },
        required: ["notes"],
    },
}

// export const NOTES_TOOL_SCHEMA = {
//     name: "formatNotes",
//     description: "Format the notes response",
//     parameters: {
//         type: "object",
//         properties: {
//             notes: {
//                 type: "array",
//                 items: {
//                     type: "object",
//                     properties: {
//                         note: {
//                         type: "string",
//                         description: "The notes",
//                         },
//                         pageNumbers: {
//                         type: "array",
//                         items: {
//                             type: "number",
//                             description: "The page number(s) of the note",
//                         },
//                         },
//                     },
//                 },
//             },
//         },
//         required: ["notes"],
//     },
// };

// export const NOTE_PROMPT = ChatPromptTemplate.fromMessages([
//     [
//         "system",
//         `Take notes the following scientific paper.
//     This is a technical paper outlining a computer science technique.
//     The goal is to be able to create a complete understanding of the paper after reading all notes.
    
//     Rules:
//     - Include specific quotes and details inside your notes.
//     - Respond with as many notes as it might take to cover the entire paper.
//     - Go into as much detail as you can, while keeping each note on a very specific part of the paper.
//     - Include notes about the results of any experiments the paper describes.
//     - Include notes about any steps to reproduce the results of the experiments.
//     - DO NOT respond with notes like: "The author discusses how well XYZ works.", instead explain what XYZ is and how it works.
    
//     Respond with a JSON array with two keys: "note" and "pageNumbers".
//     "note" will be the specific note, and pageNumbers will be an array of numbers (if the note spans more than one page).
//     Take a deep breath, and work your way through the paper step by step.`,
//     ],
//     ["human", "Paper: {paper}"],
// ]);

export const NOTE_PROMPT = PromptTemplate.fromTemplate(
    `Take notes the following scientific paper.
This is a technical paper outlining a computer science technique.
The goal is to be able to create a complete understanding of the paper after reading all notes.

Rules:
- Include specific quotes and details inside your notes.
- Respond with as many notes as it might take to cover the entire paper.
- Go into as much detail as you can, while keeping each note on a very specific part of the paper.
- Include notes about the results of any experiments the paper describes.
- Include notes about any steps to reproduce the results of the experiments.
- DO NOT respond with notes like: "The author discusses how well XYZ works.", instead explain what XYZ is and how it works.

Respond with a JSON array with two keys: "note" and "pageNumbers".
"note" will be the specific note, and pageNumbers will be an array of numbers (if the note spans more than one page).
Take a deep breath, and work your way through the paper step by step.

Paper: {paper}`
);

export type ArxivPaperNote = {
    note: string;
    pageNumbers: number[];
}

export const outputParser = (output: BaseMessageChunk): Array<ArxivPaperNote> => {
    console.log('output',output);
    const toolCalls = output.additional_kwargs.function_call;
    if (!toolCalls  || !toolCalls.arguments) {
        throw new Error("No tool calls found in output");
    }
    // formats the arguments of the tool calls
    // from: {"notes":[{"note":"The author discusses how well XYZ works.","pageNumbers":[]}]}
    // to:  [{"note":"The author discusses how well XYZ works.","pageNumbers":[]}]
    let args = toolCalls.arguments;
    args = args.substring(args.indexOf("["), args.lastIndexOf("]")+1);

    const notes: Array<ArxivPaperNote> = JSON.parse(args);
    return notes;
}