"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function ThemeToggle() {
  const { setTheme } = useTheme();

  return (
    <DropdownMenu>
      {/* Both icons are always in the DOM; the .dark class (not JS) decides
          which is visible, so this needs no client-mount guard to stay
          hydration-safe. */}
      <DropdownMenuTrigger
        render={<Button variant="outline" size="icon" className="rounded-full" />}
      >
        <Sun className="size-4 scale-100 dark:scale-0 transition-transform" />
        <Moon className="absolute size-4 scale-0 dark:scale-100 transition-transform" />
        <span className="sr-only">Toggle theme</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>Light</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>Dark</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>System</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
