import { lectures } from "@/data/lectures";
import Sidebar from "@/components/Sidebar";
import LectureContent from "@/components/LectureContent";
import { notFound } from "next/navigation";

interface PageProps {
  params: Promise<{
    lectureId: string;
  }>;
}

export async function generateStaticParams() {
  return lectures.map((lecture) => ({
    lectureId: lecture.id,
  }));
}

export async function generateMetadata({ params }: PageProps) {
  const { lectureId } = await params;
  const lecture = lectures.find((l) => l.id === lectureId);

  if (!lecture) {
    return {
      title: "Not Found",
    };
  }

  return {
    title: `${lecture.title} - AML Notes`,
    description: `Study notes for ${lecture.title}`,
  };
}

export default async function Page({ params }: PageProps) {
  const { lectureId } = await params;
  const lecture = lectures.find((l) => l.id === lectureId);

  if (!lecture) {
    notFound();
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeId={lectureId} />

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto md:ml-64">
        <div className="max-w-4xl mx-auto px-6 py-12 md:py-16">
          <LectureContent content={lecture.content} title={lecture.title} />
        </div>
      </main>
    </div>
  );
}
