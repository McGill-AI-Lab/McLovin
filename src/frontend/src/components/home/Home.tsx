import { Box, Typography, Container } from "@mui/material";
import { motion } from "framer-motion";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export const Home = () => {
  const navigate = useNavigate();
  const [showContent, setShowContent] = useState(false);

  const titleVariants = {
    hidden: { opacity: 0, y: -20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 1.2,
        ease: [0.6, -0.05, 0.01, 0.99],
      },
    },
  };

  const arrowVariants = {
    initial: { opacity: 0, y: 0 },
    animate: {
      opacity: [0, 1, 1],
      y: [0, 10, 0],
      transition: {
        duration: 2,
        repeat: showContent ? 0 : Infinity,
        repeatType: "reverse",
      },
    },
  };

  const handleArrowClick = () => {
    setShowContent(true);
    window.scrollTo({
      top: window.innerHeight,
      behavior: "smooth",
    });
  };

  const textButtonStyle = {
    color: "white",
    fontSize: "1.2rem",
    cursor: "pointer",
    position: "relative",
    display: "inline-block",
    padding: "10px 20px",
    "&::after": {
      content: '""',
      position: "absolute",
      width: "0",
      height: "2px",
      bottom: 0,
      left: "50%",
      backgroundColor: "white",
      transition: "all 0.3s ease-in-out",
    },
    "&:hover::after": {
      width: "100%",
      left: "0",
    },
  };

  return (
    <Box
      sx={{
        minHeight: "200vh",
        background: "linear-gradient(135deg, #ff9a9e, #fad0c4)",
        overflow: "hidden",
      }}
    >
      {/* First Screen */}
      <Box
        sx={{
          height: "100vh",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      >
        <Box
          sx={{
            textAlign: "center",
            position: "relative",
          }}
        >
          <motion.div
            initial="hidden"
            animate="visible"
            variants={titleVariants}
          >
            <Typography
              variant="h1"
              sx={{
                fontSize: { xs: "2.5rem", md: "4rem" },
                color: "white",
                fontWeight: 700,
                textShadow: "2px 2px 4px rgba(0,0,0,0.1)",
                mb: 4,
              }}
            >
              Find Your McGill Match
            </Typography>
          </motion.div>

          <motion.div
            initial="initial"
            animate="animate"
            variants={arrowVariants}
            onClick={handleArrowClick}
            style={{ cursor: "pointer" }}
          >
            <Box
              sx={{
                color: "white",
                fontSize: "3rem",
                "&:hover": {
                  transform: "scale(1.1)",
                  transition: "transform 0.2s ease-in-out",
                },
              }}
            >
              ↓
            </Box>
          </motion.div>
        </Box>
      </Box>

      {/* Second Screen */}
      <Box
        sx={{
          height: "100vh",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 4,
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={showContent ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <Typography
            variant="h2"
            sx={{
              fontSize: { xs: "1.8rem", md: "2.5rem" },
              color: "white",
              fontWeight: 600,
              textAlign: "center",
              mb: 6,
            }}
          >
            Welcome to McGill's AI Matchmaker!
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={showContent ? { opacity: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
          sx={{
            display: "flex",
            flexDirection: { xs: "column", sm: "row" },
            gap: 4,
          }}
        >
          <Typography onClick={() => navigate("/signup")} sx={textButtonStyle}>
            I'm new here!
          </Typography>

          <Typography onClick={() => navigate("/login")} sx={textButtonStyle}>
            I'm a returning user!
          </Typography>
        </motion.div>
      </Box>
    </Box>
  );
};
