import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { Button, Card, CardBody, CardHeader, Chip, Divider, Link } from "@heroui/react";
import {Input} from "@heroui/react";
import {Progress} from "@heroui/react";
import { login } from "../http/user.ts"

export default function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [submitError, setSubmitError] = useState("");

  const toggleLoading = () => {
    setIsLoading(!isLoading);
  }

  const usernameError = useMemo(() => {
    if (!username) return "";
    if (username.trim().length < 2) return "用户名至少 2 个字符";
    return "";
  }, [username]);

  const passwordError = useMemo(() => {
    if (!password) return "";
    if (password.length < 6) return "密码至少 6 位";
    return "";
  }, [password]);

  const isFormValid = !!username && !!password && !usernameError && !passwordError;

  const renderButton = () => {
    if(isLoading) return(
      <Progress
        isIndeterminate
        aria-label="Loading..."
        className="max-w-md"
        size="sm"
      />
    )
    return (
      <Button
      type="submit"
      color="primary"
      fullWidth
      isLoading={isLoading}
      isDisabled={!isFormValid || isLoading}
      size="lg"
    >
      登录
    </Button>
    )
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isFormValid) {
      setSubmitError("请先正确填写登录信息");
      return;
    }

    setSubmitError("");
    setIsLoading(true);
    try {
      const { data } = await login({
        username: username.trim(),
        password,
      });

      localStorage.setItem("token", data.token);
      localStorage.setItem("username", data.username);
      navigate("/");
    } catch (error) {
      const message =
        (error as { response?: { data?: { error?: string } } })?.response?.data?.error ||
        "登录失败，请稍后重试";
      setSubmitError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-100 via-cyan-50 to-blue-100 px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl items-center">
        <Card className="w-full overflow-hidden border border-white/70 shadow-2xl">
          <CardBody className="grid grid-cols-1 p-0 lg:grid-cols-2">
            <section className="hidden bg-slate-900 p-8 text-white lg:flex lg:flex-col lg:justify-between xl:p-12">
              <div className="space-y-4">
                <Chip color="primary" variant="flat" className="border border-white/20 bg-white/10 text-white">
                  HeroUI Auth
                </Chip>
                <h2 className="text-3xl font-semibold leading-tight xl:text-4xl">
                  欢迎回来，继续你的学习旅程
                </h2>
                <p className="max-w-md text-sm text-slate-200 xl:text-base">
                  自适应页面会根据分辨率自动调整布局，移动端专注表单，桌面端展示完整信息面板。
                </p>
              </div>
              <p className="text-xs text-slate-300">Secure login experience with HeroUI components.</p>
            </section>

            <section className="p-5 sm:p-8 lg:p-10 xl:p-12">
              <CardHeader className="flex flex-col items-start gap-2 p-0">
                <h1 className="text-[clamp(1.6rem,2.3vw,2.2rem)] font-bold tracking-tight text-slate-800">
                  登录账户
                </h1>
                <p className="text-sm text-default-500 sm:text-base">输入邮箱和密码，快速进入系统。</p>
              </CardHeader>

              <form onSubmit={handleSubmit} className="mt-6 space-y-4">
                <Input
                  label="用户名"
                  type="text"
                  value={username}
                  onValueChange={(value) => {
                    setUsername(value);
                    setSubmitError("");
                  }}
                  isRequired
                  autoComplete="username"
                  size="lg"
                  radius="md"
                  isInvalid={!!usernameError}
                  errorMessage={usernameError}
                />

                <Input
                  label="密码"
                  type={isPasswordVisible ? "text" : "password"}
                  value={password}
                  onValueChange={(value) => {
                    setPassword(value);
                    setSubmitError("");
                  }}
                  isRequired
                  autoComplete="current-password"
                  size="lg"
                  radius="md"
                  isInvalid={!!passwordError}
                  errorMessage={passwordError}
                  classNames={{
                    base: "data-[focus=true]:outline-none data-[focus-visible=true]:outline-none",
                    inputWrapper:
                      "shadow-none group-data-[focus=true]:shadow-none group-data-[focus=true]:border-default-300 group-data-[focus-visible=true]:ring-0 group-data-[focus-visible=true]:ring-offset-0",
                  }}
                />
                {/* <Button onClick={toggleLoading }>切换</Button> */}
                <div className="flex items-center justify-between">
                  <Link href="#" size="sm" className="text-default-500">
                    忘记密码？
                  </Link>
                  <Chip size="sm" variant="flat" color="default">
                    安全登录
                  </Chip>
                </div>

                {submitError ? <p className="text-sm text-danger">{submitError}</p> : null}
                {renderButton()}
              </form>

              <Divider className="my-6" />

              <p className="text-center text-sm text-default-500">
                还没有账户？{" "}
                <Link href="/register" size="sm">
                  立即注册
                </Link>
              </p>
            </section>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
