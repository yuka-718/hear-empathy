import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://yuka-718.github.io/hear-empathy/'),
  title: 'HearEmpathy — 声から伝わり方を整える',
  description:
    'プレゼン中の緊張度・声の熱量・話すテンポをリアルタイムで可視化し、その場で伝え方を整える音声リハーサルツール。',
  openGraph: {
    title: 'HearEmpathy — 声を、整える。',
    description:
      '声の緊張サイン・熱量・テンポをリアルタイムで可視化する音声リハーサルツール。',
    url: 'https://yuka-718.github.io/hear-empathy/',
    siteName: 'HearEmpathy',
    locale: 'ja_JP',
    type: 'website',
    images: [
      {
        url: 'https://yuka-718.github.io/hear-empathy/og.png',
        width: 1600,
        height: 900,
        alt: 'HearEmpathy — 声を、整える。',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'HearEmpathy — 声を、整える。',
    description:
      '声の緊張サイン・熱量・テンポをリアルタイムで可視化する音声リハーサルツール。',
    images: ['https://yuka-718.github.io/hear-empathy/og.png'],
  },
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
