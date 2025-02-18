import { useState } from "react";
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  InputAdornment,
  IconButton,
} from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";
import FavoriteIcon from "@mui/icons-material/Favorite";

export const Login = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [credentials, setCredentials] = useState({
    email: "",
    password: "",
  });

  return (
    <Container
      maxWidth="sm"
      className="min-h-screen flex items-center justify-center"
    >
      <Paper elevation={0} className="w-full p-8 space-y-6">
        <Box className="text-center space-y-4">
          <FavoriteIcon
            sx={{
              fontSize: 40,
              color: "primary.main",
            }}
          />
          <Typography variant="h4" className="font-semibold">
            Welcome back
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Continue your journey to find meaningful connections
          </Typography>
        </Box>

        <form className="space-y-4">
          <TextField
            fullWidth
            label="Email"
            variant="outlined"
            value={credentials.email}
            onChange={(e) =>
              setCredentials({ ...credentials, email: e.target.value })
            }
            className="bg-gray-50"
          />

          <TextField
            fullWidth
            label="Password"
            variant="outlined"
            type={showPassword ? "text" : "password"}
            value={credentials.password}
            onChange={(e) =>
              setCredentials({ ...credentials, password: e.target.value })
            }
            className="bg-gray-50"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />

          <Button fullWidth variant="contained" size="large" className="mt-6">
            Log in
          </Button>
        </form>
      </Paper>
    </Container>
  );
};
