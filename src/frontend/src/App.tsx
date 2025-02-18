import { useState } from "react";
import { ThemeProvider } from "@mui/material/styles";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import reactLogo from "./assets/react.svg";
import { Box } from "@mui/material";
import viteLogo from "/vite.svg";
import "./App.css";
import { theme } from "./theme/theme";
import { Layout } from "./components/layout/Layout";
import { Login } from "./components/auth/Login";
import { Dashboard } from "./components/dashboard/Dashboard";
import { Profile } from "./components/profile/Profile";
import { Home } from "./components/home/Home";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        <Box
          sx={{
            width: "100%",
            minHeight: "100vh",
            overflow: "hidden",
            margin: 0,
            padding: 0,
          }}
        >
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />
            {/* Protected routes within Layout */}
            <Route path="/" element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="profile" element={<Profile />} />
            </Route>
          </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
