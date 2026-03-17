import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { Button, Card, CardBody, CardHeader, Chip, Divider, Input, Link } from "@heroui/react";
import { register } from "../http/user"

export default function RegisterPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isConfirmPasswordVisible, setIsConfirmPasswordVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [submitError, setSubmitError] = useState("");

  const nameError = useMemo(() => {
    if (!username) return "";
    if (username.trim().length < 2) return "用户名至少 2 个字符";
    return "";
  }, [username]);

  const emailError = useMemo(() => {
    if (!email) return "";
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) return "请输入有效的邮箱地址";
    return "";
  }, [email]);

  const passwordStrength = useMemo(() => {
    if (!password) return { text: "未设置", color: "default" as const };
    if (password.length < 8) return { text: "弱", color: "danger" as const };
    const hasLetter = /[a-zA-Z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSymbol = /[^a-zA-Z0-9]/.test(password);
    const score = [hasLetter, hasNumber, hasSymbol].filter(Boolean).length;

    if (score <= 1) return { text: "弱", color: "danger" as const };
    if (score === 2) return { text: "中", color: "warning" as const };
    return { text: "强", color: "success" as const };
  }, [password]);

  const passwordError = useMemo(() => {
    if (!password) return "";
    if (password.length < 6) return "密码至少 6 位";
    // if (!/[a-zA-Z]/.test(password) || !/\d/.test(password)) {
    //   return "建议包含字母和数字";
    // }
    return "";
  }, [password]);

  const confirmPasswordError = useMemo(() => {
    if (!confirmPassword) return "";
    if (password !== confirmPassword) return "两次输入的密码不一致";
    return "";
  }, [password, confirmPassword]);

  const isFormValid = useMemo(() => {
    return (
      username.trim().length >= 2 &&
      !!email &&
      !emailError &&
      !!password &&
      !passwordError &&
      !!confirmPassword &&
      !confirmPasswordError
    );
  }, [username, email, emailError, password, passwordError, confirmPassword, confirmPasswordError]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!isFormValid) {
      setSubmitError("请先完善表单信息");
      return;
    }

    setSubmitError("");
    setIsLoading(true);

    // TODO: 替换为实际注册逻辑
    try{
      const { data } = await register({
        username: username.trim(),
        password,
        email
      });

      localStorage.setItem("token", data.token);
      localStorage.setItem("username", data.username);
      navigate("/");
    }catch (error){
      const message =
        (error as { response?: { data?: { error?: string } } })?.response?.data?.error ||
        "注册失败，请稍后重试";
      setSubmitError(message);
    }
    setIsLoading(false);

    navigate("/login");
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-slate-100 to-blue-100 px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-6xl items-center">
        <Card className="w-full overflow-hidden border border-white/70 shadow-2xl">
          <CardBody className="grid grid-cols-1 p-0 lg:grid-cols-2">
            <section className="hidden bg-blue-900 p-8 text-white lg:flex lg:flex-col lg:justify-between xl:p-12">
              <div className="space-y-4">
                <Chip color="primary" variant="flat" className="border border-white/25 bg-white/10 text-white">
                  Create Account
                </Chip>
                <h2 className="text-3xl font-semibold leading-tight xl:text-4xl">
                  创建你的学习空间
                </h2>
                <p className="max-w-md text-sm text-blue-100 xl:text-base">
                  页面会在不同分辨率下自动切换布局，移动端聚焦输入，桌面端提供额外引导信息。
                </p>
              </div>
              <p className="text-xs text-blue-200">Responsive auth experience built with HeroUI + Tailwind.</p>
            </section>

            <section className="p-5 sm:p-8 lg:p-10 xl:p-12">
              <CardHeader className="flex flex-col items-start gap-2 p-0">
                <h1 className="text-[clamp(1.6rem,2.3vw,2.2rem)] font-bold tracking-tight text-slate-800">
                  创建账户
                </h1>
                <p className="text-sm text-default-500 sm:text-base">填写资料，快速完成注册。</p>
              </CardHeader>

              <form onSubmit={handleSubmit} className="mt-6 space-y-4">
                <Input
                  label="用户名"
                  placeholder="请输入用户名"
                  value={username}
                  onValueChange={(v) => {
                    setUsername(v);
                    setSubmitError("");
                  }}
                  isRequired
                  autoComplete="name"
                  size="lg"
                  radius="md"
                  isInvalid={!!nameError}
                  errorMessage={nameError}
                />

                <Input
                  label="邮箱"
                  placeholder="you@example.com"
                  type="email"
                  value={email}
                  onValueChange={(v) => {
                    setEmail(v);
                    setSubmitError("");
                  }}
                  isRequired
                  autoComplete="email"
                  size="lg"
                  radius="md"
                  isInvalid={!!emailError}
                  errorMessage={emailError}
                />

                <div className="space-y-2">
                  <Input
                    label="密码"
                    placeholder="至少 6 位，建议含字母和数字"
                    type={isPasswordVisible ? "text" : "password"}
                    value={password}
                    onValueChange={(v) => {
                      setPassword(v);
                      setSubmitError("");
                    }}
                    isRequired
                    minLength={6}
                    autoComplete="new-password"
                    size="lg"
                    radius="md"
                    isInvalid={!!passwordError}
                    errorMessage={passwordError}
                  />
                  <div className="flex items-center justify-between text-xs text-default-500">
                    <span>密码强度</span>
                    <Chip size="sm" variant="flat" color={passwordStrength.color}>
                      {passwordStrength.text}
                    </Chip>
                  </div>
                </div>

                <Input
                  label="确认密码"
                  placeholder="请再次输入密码"
                  type={isConfirmPasswordVisible ? "text" : "password"}
                  value={confirmPassword}
                  onValueChange={(v) => {
                    setConfirmPassword(v);
                    setSubmitError("");
                  }}
                  isRequired
                  autoComplete="new-password"
                  size="lg"
                  radius="md"
                  isInvalid={!!confirmPasswordError}
                  errorMessage={confirmPasswordError}
                />

                {submitError ? <p className="text-sm text-danger">{submitError}</p> : null}

                <Button type="submit" color="primary" fullWidth isLoading={isLoading} isDisabled={!isFormValid || isLoading} size="lg">
                  注册
                </Button>
              </form>

              <Divider className="my-6" />

              <p className="text-center text-sm text-default-500">
                已有账户？{" "}
                <Link href="/login" size="sm">
                  立即登录
                </Link>
              </p>
            </section>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
