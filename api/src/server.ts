import express from "express";
import { takeNotes } from "notes/index.js";
function main() {
    const app = express();
    const port = 8000;
    app.use(express.json());

    app.get("/", (_req, res) => {
        // health check
        res.status(200).send("ok");
    })

    app.post("/take_notes", async (req, res) => {
        const { paperUrl, name, pagesToDelete } = req.body;
        const notes = await takeNotes({ paperUrl, name, pagesToDelete });
        res.status(200).send(notes);
        return;
    })

    app.listen(port, () => {
        console.log(`Server started. Listening on port ${port}`);
    });
}
main();