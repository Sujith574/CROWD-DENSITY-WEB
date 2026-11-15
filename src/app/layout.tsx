import type { Metadata } from 'next';
import Script from 'next/script';
import './globals.css';

export const metadata: Metadata = {
  title: 'Crowd Density Monitor | CSRNet Hotspot Mapping',
  description:
    'Real-time crowd density estimation and congestion hotspot mapping using Dilated CNNs (CSRNet) - 100% client-side ONNX inference, zero server cost.',
  keywords: ['crowd density', 'CSRNet', 'ONNX', 'hotspot', 'congestion', 'AI'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <Script
          src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"
          strategy="beforeInteractive"
        />
      </head>
      <body className="antialiased bg-slate-950 text-slate-200 min-h-screen">
        {children}
      </body>
    </html>
  );
}
