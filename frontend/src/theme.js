import { extendTheme } from "@chakra-ui/react";

export const theme = extendTheme({
  fonts: {
    heading: "Inter, system-ui, sans-serif",
    body: "Inter, system-ui, sans-serif",
  },
  colors: {
    ink: "#18201d",
    sage: { 50: "#f3f7f2", 100: "#e4efe5", 500: "#4e765e", 700: "#31523f" },
    coral: { 50: "#fff4ef", 200: "#ffd7c8", 500: "#d76542" },
  },
  styles: {
    global: {
      "html, body": { background: "#f7f8f5", color: "#18201d" },
      body: { minWidth: "320px" },
    },
  },
  components: {
    Button: { baseStyle: { borderRadius: "full", fontWeight: 600 } },
    Input: { defaultProps: { focusBorderColor: "sage.500" } },
    Textarea: { defaultProps: { focusBorderColor: "sage.500" } },
  },
});

