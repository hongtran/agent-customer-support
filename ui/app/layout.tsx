import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CenLab Agent — Test UI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen bg-white antialiased">{children}</body>
    </html>
  );
}
