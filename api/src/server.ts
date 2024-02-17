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

async function loadFromUrl(url: string): Promise<Buffer> {
    const response = await axios.get(url, {
        responseType: "arraybuffer",
    });
    return response.data;
}

async function deletePages(pdf: Buffer, pagesToDelete: number[]): Promise<Buffer> {
    const pdfDoc = await PDFDocument.load(pdf);
    let numToOffsetBy = 0;
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
        baseUrl: "https://23f4-107-167-188-205.ngrok-free.app/",
        model: "mixtral",
        temperature: 0.0,
    });
    const modelWithTool = model.bind({
        functions: [NOTES_TOOL_SCHEMA],
    });
    const chain  = NOTE_PROMPT.pipe(modelWithTool).pipe(outputParser);
    // console.log(documentsAsString);
    const response = await chain.invoke({
        paper: documentsAsString, // {paper} in NOTE_PROMPT
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
    if (!paperUrl.endsWith(".pdf")) {
        throw new Error("Unsupported file type");
    }
    let pdfAsBuffer = await loadFromUrl(paperUrl);
    if (pagesToDelete && pagesToDelete.length > 0) {
        pdfAsBuffer = await deletePages(pdfAsBuffer, pagesToDelete);
    }
    const documents = await convertPdfToDocument(pdfAsBuffer);
    const notes = await generateNotes(documents);
    console.log(notes);
    console.log("length", notes.length);

    // const model = new ChatOllama({
    //     baseUrl: "https://2d68-34-139-179-102.ngrok-free.app", // Default value
    //     model: "mixtral", // Default value
    //   });
      
    //   const stream = await model
    //     .pipe(new StringOutputParser())
    //     .stream(`Translate "I love programming" into German.`);
      
    //   const chunks = [];
    //   for await (const chunk of stream) {
    //     chunks.push(chunk);
    //   }
      
    //   console.log(chunks.join(""));
}
// https://arxiv.org/pdf/2305.15334.pdf Gorilla paper
main({ paperUrl: "https://arxiv.org/pdf/2402.10087.pdf", name: "test", pagesToDelete: [] })