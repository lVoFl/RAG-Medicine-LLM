import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & {
  size?: number;
  height?: number;
  width?: number;
};

export const Edit = ({size, height, width, ...props}: IconProps) => {
  return (
    <svg
      viewBox="0 0 1024 1024"
      width={size || width || 24}
      height={size || height || 24}
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M766.88 435.264l-176.608-176.64 66.72-66.752 176.544 176.704-66.656 66.688zM401.44 800.96L224.64 624.192l0.256 0.064L545.024 303.904l176.64 176.64L401.376 800.96zM224 801.92v-87.872l87.712 87.68-87.68 0.192z m655.04-478.528l-176.768-176.736A60.96 60.96 0 0 0 656.96 128a63.968 63.968 0 0 0-45.12 18.848L179.584 579.008a63.936 63.936 0 0 0-17.92 54.368c-0.768 2.688-1.696 5.312-1.696 8.256v160.288c0 35.136 28.576 63.68 63.712 63.68h160.32c2.88 0 5.504-0.896 8.192-1.632 2.976 0.416 5.952 0.832 8.96 0.832 16.416 0 32.896-6.272 45.44-18.816l432.16-432.16a64 64 0 0 0 0.224-90.432z" />
    </svg>
  );
};

export const Trash = ({size, height, width, ...props}: IconProps) => {
  return (
    <svg
      viewBox="0 0 1024 1024"
      width={size || width || 24}
      height={size || height || 24}
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
  >
    <path fill="currentColor" d="M736 352.032L736.096 800h-0.128L288 799.968 288.032 352 736 352.032zM384 224h256v64h-256V224z m448 64h-128V202.624C704 182.048 687.232 160 640.16 160h-256.32C336.768 160 320 182.048 320 202.624V288H192a32 32 0 1 0 0 64h32V799.968C224 835.296 252.704 864 288.032 864h447.936A64.064 64.064 0 0 0 800 799.968V352h32a32 32 0 1 0 0-64z"/>
    <path fill="currentColor" d="M608 690.56a32 32 0 0 0 32-32V448a32 32 0 1 0-64 0v210.56a32 32 0 0 0 32 32M416 690.56a32 32 0 0 0 32-32V448a32 32 0 1 0-64 0v210.56a32 32 0 0 0 32 32"/>
  </svg>
  );
};