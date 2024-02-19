import axios from "axios";
import { PDFDocument } from 'pdf-lib';
import { Document } from "langchain/document";
import { writeFile, unlink } from "fs/promises";
import { UnstructuredLoader } from "langchain/document_loaders/fs/unstructured";
import { formatDocumentsAsString } from "langchain/util/document";
import { OllamaFunctions } from "langchain/experimental/chat_models/ollama_functions";
import { NOTES_TOOL_SCHEMA, 
    NOTE_PROMPT, 
    outputParser, 
    ArxivPaperNote,
    toolSystemPromptTemplate } from "./prompts.js";
import { HumanMessage } from "langchain/schema";
import { ChatOllama } from "langchain/chat_models/ollama";
import { StringOutputParser } from "langchain/schema/output_parser";
import { ChatPromptTemplate, PromptTemplate } from "langchain/prompts";
import { JsonOutputFunctionsParser } from "langchain/output_parsers";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

async function loadFromUrl(url: string): Promise<Buffer> {
    const response = await axios.get(url, {
        responseType: "arraybuffer",
    });
    return response.data;
}

async function deletePages(pdf: Buffer, pagesToDelete: number[]): Promise<Buffer> {
    const pdfDoc = await PDFDocument.load(pdf);
    let numToOffsetBy = 1;
    for (const page of pagesToDelete) {
        pdfDoc.removePage(page - numToOffsetBy);
        numToOffsetBy++;
    }
    const pdfBytes = await pdfDoc.save();
    return Buffer.from(pdfBytes);
}

async function convertPdfToDocument(pdf: Buffer): Promise<Array<Document>> {
    if (!process.env.UNSTRUCTURED_API_KEY) {
        throw new Error("UNSTRUCTURED_API_KEY is not set");
    }
    const randomName = Math.random().toString(36).substring(7);
    await writeFile(`pdfs/${randomName}.pdf`, pdf);
    const loader = new UnstructuredLoader(`pdfs/${randomName}.pdf`, {
        apiKey: process.env.UNSTRUCTURED_API_KEY,
        strategy: "hi_res",
    });
    const documents = await loader.load();
    await unlink(`pdfs/${randomName}.pdf`);
    return documents;
}

async function generateNotes(documents:Array<Document>): Promise<Array<ArxivPaperNote>> {
    const documentsAsString = formatDocumentsAsString(documents);
    const model = new OllamaFunctions({
        baseUrl: "http://localhost:11434",
        model: "mistrallite",
        temperature: 0.0,
        // toolSystemPromptTemplate: toolSystemPromptTemplate,
    });
    const modelWithTool = model.bind({
        functions: [NOTES_TOOL_SCHEMA],
        function_call: {
            name: "formatNotes",
            
        },
    });
    const chain  = NOTE_PROMPT.pipe(modelWithTool).pipe(outputParser);
    // console.log(documentsAsString);
    const testDoc = `In this paper, we explore the use of self-instruct fine-tuning and retrieval to enable LLMs to accurately select from a large, overlapping, and changing set tools expressed using their APIs and API
    documentation. We construct, APIBench, a large corpus of APIs with complex and often overlapping
    functionality by scraping ML APIs (models) from public model hubs. We choose three major model
    hubs for dataset construction: TorchHub, TensorHub and HuggingFace. We exhaustively include
    every API call in TorchHub (94 API calls) and TensorHub (696 API calls); For HuggingFace, since
    the models come in a large number and lots of the models don’t have a specification, we choose the
    most downloaded 20 models per task category (in a total of 925). We also generate 10 synthetic
    user question prompts per API using Self-Instruct [42]. Thus, each entry in the dataset becomes an
    instruction reference API pair. We adopt a common AST sub-tree matching technique to evaluate the
    functional correctness of the generated API. We first parse the generated code into an AST tree, then
    find a sub-tree whose root node is the API call that we care about (e.g., torch.hub.load) and use it
    to index our dataset. We check the functional correctness and hallucination problem for the LLMs,
    reporting the corresponding accuracy`;
    const response = await chain.invoke({
        paper: testDoc, // {paper} in NOTE_PROMPT
    });
    return response;
}

async function main({
    paperUrl,
    name,
    pagesToDelete
}: {
    paperUrl: string,
    name: string,
    pagesToDelete: number[]
}) {
    // Loads the paper and processes it using unstructured API
    // if (!paperUrl.endsWith(".pdf")) {
    //     throw new Error("Unsupported file type");
    // }
    // let pdfAsBuffer = await loadFromUrl(paperUrl);
    // if (pagesToDelete && pagesToDelete.length > 0) {
    //     pdfAsBuffer = await deletePages(pdfAsBuffer, pagesToDelete);
    // }
    // const documents = await convertPdfToDocument(pdfAsBuffer);
    // console.log(documents);
    // console.log("length", documents.length);

    //For now just using empty array and testDoc inside generateNotes as it takes the llm too long to process the full paper
    const notes = await generateNotes(new Array<Document>());
    console.log(notes);
    console.log("length", notes.length);

    //------------------------------------
    // const EXTRACTION_TEMPLATE = `Extract and save the relevant entities mentioned in the following passage together with their properties.

    // Passage:
    // {input}
    // `;

    // const prompt = PromptTemplate.fromTemplate(EXTRACTION_TEMPLATE);

    // // Use Zod for easier schema declaration
    // const schema = z.object({
    // people: z.array(
    //     z.object({
    //     name: z.string().describe("The name of a person"),
    //     height: z.number().describe("The person's height"),
    //     hairColor: z.optional(z.string()).describe("The person's hair color"),
    //     })
    // ),
    // });
    // const model = new OllamaFunctions({
    //     baseUrl: "http://localhost:11434",
    //     temperature: 0.0,
    //     model: "mistrallite",
    //   }).bind({
    //     functions: [
    //         {
    //         name: "information_extraction",
    //         description: "Extracts the relevant information from the passage.",
    //         parameters: {
    //             type: "object",
    //             properties: zodToJsonSchema(schema),
    //         },
    //         },
    //     ],
    //     function_call: {
    //         name: "information_extraction",
    //     },
    // });

    // // Use a JsonOutputFunctionsParser to get the parsed JSON response directly.
    // const chain = await prompt.pipe(model).pipe(new JsonOutputFunctionsParser());

    // const response = await chain.invoke({
    // input:
    //     "Alex is 5 feet tall. Claudia is 1 foot taller than Alex and jumps higher than him. Claudia has orange hair and Alex is blonde.",
    // });

    // console.log(response);
    //------------------------------------
}

const text = `[]`.replace(/api:start:/g, '');
    
// https://arxiv.org/pdf/2305.15334.pdf Gorilla paper
main({ paperUrl: "https://arxiv.org/pdf/2305.15334.pdf", name: "test", pagesToDelete: [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] })