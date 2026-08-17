import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CenLab Support Agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen bg-white antialiased">{children}</body>
    </html>
  );
}
