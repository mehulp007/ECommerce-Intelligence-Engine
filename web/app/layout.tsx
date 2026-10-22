import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { Providers } from "@/components/providers";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const viewport: Viewport = { themeColor: "#0b0d14", colorScheme: "dark" };

export const metadata: Metadata = {
  title: { default: "E-Commerce Intelligence Engine", template: "%s · Intelligence Engine" },
  description:
    "Recommendations, visual search, churn, lifetime value, campaigns, and an analyst copilot over 31.8M fashion purchases.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
      <body className="bg-bg font-sans text-fg">
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
