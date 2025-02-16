import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    primary: {
      main: "#ff6f61",
      light: "#ff9a9e",
      dark: "#e55b4c",
    },
    background: {
      default: "#f4f4f9",
    },
  },
  typography: {
    fontFamily: "Arial, sans-serif",
  },
});
