import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router";

const LoginPage = lazy(() => import("./pages/LoginPage"));
const RegisterPage = lazy(() => import("./pages/RegisterPage"));
const HomePage = lazy(() => import("./pages/HomePage"));
const KnowledgeManagePage = lazy(() => import("./pages/KnowledgeManagePage"));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div className="flex min-h-screen items-center justify-center text-slate-500">加载中...</div>}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={<HomePage />} />
          <Route path="/admin/knowledge" element={<KnowledgeManagePage />} />
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
