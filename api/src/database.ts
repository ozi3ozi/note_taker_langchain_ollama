import { Document } from "langchain/document";
import { SupabaseClient, createClient } from "@supabase/supabase-js";
import { Database, Json } from "generated.js";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OllamaEmbeddings } from "langchain/embeddings/ollama";
import { ArxivPaperNote } from "notes/prompts.js";

// Tables and function names in generated.ts
// Were not put in generated.ts because that file in generated during build. See api/package.json
export const ARXIV_EMBEDDINGS_TABLE = "arxiv_embeddings";
export const ARXIV_PAPERS_TABLE = "arxiv_papers";
export const ARXIV_QUESTION_ANSWERING_TABLE = "arxiv_question_answering";
export const ARXIV_SEARCH_FUNCTION = "match_documents";

export class SupabaseDatabase {
    vectoreStore: SupabaseVectorStore;
    client: SupabaseClient<Database, "public", any>;

    constructor(
        vectoreStore: SupabaseVectorStore,
        client: SupabaseClient<Database, "public", any>
    ) {
        this.vectoreStore = vectoreStore;
        this.client = client;
    }

    //Instantiate a SupabaseDatabase from Array<Document>
    static async fromDocuments(documents: Array<Document>): Promise<SupabaseDatabase> {
        const privateKey = process.env.SUPABASE_PRIVATE_KEY;
        const supabaseUrl = process.env.SUPABASE_URL;
        if (!privateKey || !supabaseUrl) {
            throw new Error("SUPABASE_URL or SUPABASE_PRIVATE_KEY is not set");
        }

        const embeddings = new OllamaEmbeddings({
            model: "mistrallite", 
            baseUrl: "http://localhost:11434", // default value
          });

        const supabase = createClient<Database>(supabaseUrl, privateKey);
        const vectorStore = await SupabaseVectorStore.fromDocuments(
            documents,
            embeddings,
            {
                client: supabase,
                tableName: ARXIV_EMBEDDINGS_TABLE,
                queryName: ARXIV_SEARCH_FUNCTION,
            }
        );

        return new this(vectorStore, supabase);
    }

    async addPaper({
        paperUrl,
        name,
        notes,
        paper,
    }:{
        paperUrl: string,
        name: string,
        notes: Array<ArxivPaperNote>,
        paper: string
    }) {
        const { data, error } = await this.client
            .from(ARXIV_PAPERS_TABLE)
            .insert({ // Make sure column order is the same as in generated.ts
                arxiv_url: paperUrl,
                name,
                notes: notes as Json,
                paper,
        }).select();
        if (error) {
            throw error;
        }

        console.log("Data added:", data);
    }
}