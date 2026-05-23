import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
});

async function main() {
    const session = await ai.live.connect({
        model: "models/gemini-2.5-flash-preview",

        callbacks: {
            onopen: () => {
                console.log("Connected to Gemini Live");
            },

            onmessage: (message) => {
                console.dir(message, { depth: null });
            },

            onerror: (error) => {
                console.error(error);
            },

            onclose: (event) => {
                console.log("Connection closed", event.reason);
            },
        },

        config: {
            responseModalities: ["TEXT"],
        },
    });

    session.sendClientContent({
        turns: [
            {
                role: "user",
                parts: [{ text: "Hello Gemini!" }],
            },
        ],
        turnComplete: true,
    });
}

main();