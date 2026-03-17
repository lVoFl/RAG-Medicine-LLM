import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
// import searchRouter from "./routes/search.js";
// import chatRouter from "./routes/chat.js";
import authRouter from "./routes/auth.js";
import conversationsRouter from "./routes/conversations.js";
import messagesRouter from "./routes/messages.js";
import { errorHandler } from "./middleware/errorHandler.js";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

// app.use("/api/search", searchRouter);
// app.use("/api/chat", chatRouter);
app.use("/api/auth", authRouter);
app.use("/api/conversations", conversationsRouter);
app.use("/api/conversations", messagesRouter);

app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
