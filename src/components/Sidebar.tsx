"use client";

import Link from "next/link";
import { lectures_index } from "@/data/lectures";
import { useState } from "react";

interface SidebarProps {
  activeId: string;
}

export default function Sidebar({ activeId }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <>
      {/* Mobile Menu Toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-50 md:hidden bg-blue-600 text-white p-2 rounded-lg"
      >
        {isOpen ? "✕" : "☰"}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-screen w-64 bg-gradient-to-b from-blue-900 to-blue-800 text-white overflow-y-auto pt-4 md:flex flex-col transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0`}
      >
        <div className="px-6 py-4 border-b border-blue-700">
          <h1 className="text-2xl font-bold">AML Notes</h1>
          <p className="text-xs text-blue-200 mt-2">by Kartikey Mishra</p>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          {lectures_index.map((lecture) => (
            <Link
              key={lecture.id}
              href={`/${lecture.id}`}
              onClick={() => setIsOpen(false)}
              className={`block px-4 py-3 rounded-lg transition-all duration-200 ${
                activeId === lecture.id
                  ? "bg-blue-500 text-white font-semibold shadow-lg"
                  : "text-blue-100 hover:bg-blue-700"
              }`}
            >
              <div className="text-sm">
                {lecture.number > 0 && (
                  <span className="font-bold text-xs bg-blue-600 px-2 py-1 rounded mr-2">
                    L{lecture.number}
                  </span>
                )}
                {lecture.number === 0 && (
                  <span className="font-bold text-xs bg-yellow-600 px-2 py-1 rounded mr-2">
                    ⭐
                  </span>
                )}
                <span>{lecture.title}</span>
              </div>
            </Link>
          ))}
        </nav>

        <div className="px-6 py-4 border-t border-blue-700 text-xs text-blue-200">
          <p>📚 Applied Machine Learning</p>
          <p>Complete Study Material</p>
        </div>
      </aside>

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 md:hidden z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
